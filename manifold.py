"""
File containing all the manifold we are going to use for the experiments:
- Regression manifold
- Linearized regression manifold
- cross entropy manifold
- linearized cross entropy manifold
"""

import torch
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap
from functools import partial
from functorch_utils import get_params_structure, stack_gradient, custum_hvp, stack_gradient2, stack_gradient3
from torch.distributions import Normal
from torch import nn
import numpy as np
import math
import time
import copy


class regression_manifold:
    def __init__(self, model, X, y, batching=False, lambda_reg=None, noise_var=1, device="cpu"):
        self.model = model

        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.noise_var = noise_var
        self.device = device

        assert y is None if batching else True, "If batching is True, y should be None"

        # these are just to avoid to have to pass them as input of the functions
        # we use to compute the gradient and the hessian vector product
        self.fmodel = None
        self.buffers = None

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation
        """
        w_norm = sum([sum(w.view(-1) ** 2) for w in param])
        return self.lambda_reg * w_norm

    def mse_loss(self, param, data):
        # let's try to keep the prediction here without
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        # defining a model
        if self.model is None:
            raise NotImplementedError("Compute usual prediction still have to be implemented")
        else:
            y_pred = self.fmodel(param, self.buffers, x)

        criterion = torch.nn.MSELoss(reduction="sum")

        return (1.0 / self.noise_var) * criterion(y_pred, y) * 0.5

    def compute_grad_data_fitting_term(self, params, data):
        ft_compute_grad = grad(self.mse_loss)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params, data)
        return ft_per_sample_grads

    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # check if they are numpy array or torch.Tensor
        # here we need torch tensor to perform these operations
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float().to(self.device)

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float().to(self.device)

        # I would expect both current point and velocity to be
        # two vectors of shape n_params
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        self.model.zero_grad()
        # now I have to call the functorch function
        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        # and I have to reshape the velocity into being the same structure as the params
        vel_as_params = get_params_structure(velocity, params)

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        grad_data_fitting_term = 0
        if batchify:
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            grad_per_example = self.compute_grad_data_fitting_term(params, data)
            grad_per_example = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = grad_per_example.view(-1, 1)

        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg
        if batchify:
            hvp = 0
            for batch_img, batch_label in self.X:
                _, result = custum_hvp(
                    self.mse_loss,
                    (params, (batch_img, batch_label)),
                    (vel_as_params, (torch.zeros_like(batch_img), torch.zeros_like(batch_label))),
                )
                hvp += torch.cat([sub_prod.flatten() for sub_prod in result])
        else:
            _, result = custum_hvp(
                self.mse_loss, (params, data), (vel_as_params, (torch.zeros_like(data[0]), torch.zeros_like(data[1])))
            )
            hvp = [sub_prod.flatten() for sub_prod in result]
            hvp = torch.cat(hvp)

        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            hvp = hvp + hvp_reg.view(-1)

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ hvp)).flatten()

        if return_hvp:
            return second_derivative.view(-1, 1).detach().numpy(), hvp.view(-1, 1).detach().numpy()
        else:
            return second_derivative.view(-1, 1).detach().numpy()

    def get_gradient_value_in_specific_point(self, current_point):
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float()

        current_point = current_point.to(self.device)
        assert (
            len(current_point) == self.n_params
        ), "You are passing a larger vector than the number of weights in the model"

        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        self.model.zero_grad()
        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        grad_data_fitting_term = 0
        if batchify:
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            grad_per_example = self.compute_grad_data_fitting_term(params, data)
            grad_per_example = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = grad_per_example.view(-1, 1)

        if self.lambda_reg is not None:
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg
        tot_gradient = tot_gradient.to(self.device)

        return tot_gradient


# linearized regression manifold
class linearized_regression_manifold:
    def __init__(
        self, model, X, y, f_MAP, J_f_MAP, theta_MAP, batching=False, lambda_reg=None, noise_var=1, device="cpu"
    ):
        self.model = model

        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        self.batching = batching
        self.lambda_reg = lambda_reg
        self.noise_var = noise_var
        self.device = device
        self.f_MAP = f_MAP
        self.J_f_MAP = J_f_MAP
        self.theta_MAP = theta_MAP

        assert y is None if batching else True, "If batching is True, y should be None"

        # these are just to avoid to have to pass them as input of the functions
        # we use to compute the gradient and the hessian vector product
        self.fmodel = None
        self.buffers = None

        self.fmodel_map = None
        self.params_map = None
        self.buffers_map = None

    @staticmethod
    def is_diagonal():
        return False

    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation
        """
        w_norm = sum([sum(w.view(-1) ** 2) for w in param])
        return self.lambda_reg * w_norm

    def mse_loss(self, param, data, f_MAP):
        def predict(params, datas):
            y_preds = self.fmodel_map(params, self.buffers_map, datas)
            return y_preds

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        # jvp computation
        params_map = get_params_structure(self.theta_MAP, param)
        diff_weights = []
        for i in range(len(param)):
            diff_weights.append(param[i] - params_map[i])
        diff_weights = tuple(diff_weights)

        _, jvp_value = jvp(predict, (self.params_map, x), (diff_weights, torch.zeros_like(x)), strict=False)

        y_pred = f_MAP + jvp_value

        criterion = torch.nn.MSELoss(reduction="sum")

        return (1.0 / self.noise_var) * criterion(y_pred, y) * 0.5

    def compute_grad(self, params, data, f_MAP):
        ft_compute_grad = grad(self.mse_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))

        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        gradw = ft_compute_sample_grad(params, data, f_MAP)
        return gradw

    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # check if they are numpy array or torch.Tensor
        # here we need torch tensor to perform these operations
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float().to(self.device)

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float().to(self.device)

        # two vectors of shape n_params
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True

        else:
            batchify = False
            data = (self.X, self.y)

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        grad_data_fitting_term = 0
        if batchify:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = self.compute_grad(params, (batch_img, batch_label), batch_MAP)
                grad_data_fitting_term += stack_gradient2(grad_per_example, self.n_params).view(-1, 1)

        else:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            data = (self.X, self.y)

            grad_per_example = self.compute_grad(params, data, self.f_MAP)

            gradw = stack_gradient2(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)

        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        # now I have also to compute the Hvp between hessian and velocity

        vel_as_params = get_params_structure(velocity, params)

        if batchify:
            hvp = 0

            for batch_img, batch_label, batch_f_MAP in self.X:
                _, result = custum_hvp(
                    self.mse_loss,
                    (params, (batch_img, batch_label), batch_f_MAP),
                    (
                        vel_as_params,
                        (torch.zeros_like(batch_img), torch.zeros_like(batch_label)),
                        torch.zeros_like(batch_f_MAP),
                    ),
                )

                hvp += torch.cat([sub_prod.flatten() for sub_prod in result])
        else:
            _, result = custum_hvp(
                self.mse_loss,
                (params, data, self.f_MAP),
                (vel_as_params, (torch.zeros_like(data[0]), torch.zeros_like(data[1])), torch.zeros_like(self.f_MAP)),
            )

            hvp = [sub_prod.flatten() for sub_prod in result]
            hvp = torch.cat(hvp)

        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity
            tot_hvp = hvp + hvp_reg.view(-1)
        else:
            tot_hvp = hvp

        tot_gradient = tot_gradient.detach()
        tot_hvp = tot_hvp.detach()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.view(-1, 1).detach().numpy(), tot_hvp.view(-1, 1).detach().numpy()
        else:
            return second_derivative.view(-1, 1).detach().numpy()

    def get_gradient_value_in_specific_point(self, current_point):
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        if self.fmodel_map is None:
            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

        # method to return the gradient of the loss in a specific point
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float()

        current_point = current_point.to(self.device)
        assert (
            len(current_point) == self.n_params
        ), "You are passing a larger vector than the number of weights in the model"

        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        grad_data_fitting_term = 0
        if batchify:
            gradw = 0
            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = self.compute_grad(params, (batch_img, batch_label), batch_MAP)
                grad_data_fitting_term += stack_gradient2(grad_per_example, self.n_params).view(-1, 1)
        else:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            data = (self.X, self.y)
            grad_per_example = self.compute_grad(params, data, self.f_MAP)

            gradw = stack_gradient2(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)

        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        tot_gradient = tot_gradient.to(self.device)

        return tot_gradient


class cross_entropy_manifold:
    """
    Also in this case I have to split the gradient loss computation and the gradient of the regularization
    term.
    This is needed to get the correct gradient and hessian computation when using batches.
    """

    def __init__(
        self, model, X, y, batching=False, device="cpu", lambda_reg=None, type="fc", N=None, B1=None, B2=None
    ):
        self.model = model

        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        self.batching = batching
        self.device = device
        self.type = type
        # self.prior_precision = prior_precision
        self.lambda_reg = lambda_reg
        assert y is None if batching else True, "If batching is True, y should be None"

        # these are just to avoid to have to pass them as input of the functions
        # we use to compute the gradient and the hessian vector product
        self.fmodel = None
        self.buffers = None

        ## stuff we need when using barches
        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = None

        # here I can already compute the factor_loss
        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor = self.factor * (B2 / B1)

    @staticmethod
    def is_diagonal():
        return False

    def CE_loss(self, param, data):
        """
        Data fitting term of the loss
        """
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        if self.type != "fc":
            # assuming input for con
            x = x.unsqueeze(1)

        if self.model is None:
            raise NotImplementedError("Compute usual prediction still have to be implemented")
        else:
            # self.fmodel.eval()
            y_pred = self.fmodel(param, self.buffers, x)

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if self.type == "fc":
            if self.factor is not None:
                return self.factor * criterion(y_pred, y)
            else:
                return criterion(y_pred, y)
        else:
            if self.factor is not None:
                return self.factor * criterion(y_pred.view(-1), y)
            else:
                return criterion(y_pred.view(-1), y)

    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation
        """
        w_norm = sum([sum(w.view(-1) ** 2) for w in param])
        return self.lambda_reg * w_norm

    def compute_grad_data_fitting_term(self, params, data):
        # TODO: understand how to make vmap work without passing the data
        ft_compute_grad = grad(self.CE_loss)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params, data)
        return ft_per_sample_grads

    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)

        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # check if they are numpy array or torch.Tensor
        # here we need torch tensor to perform these operations
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float().to(self.device)

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float().to(self.device)

        # I would expect both current point and velocity to be
        # two vectors of shape n_params
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        # and I have to reshape the velocity into being the same structure as the params
        vel_as_params = get_params_structure(velocity, params)

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        start = time.time()
        grad_data_fitting_term = 0
        if batchify:
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            grad_per_example = self.compute_grad_data_fitting_term(params, data)
            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)
        end = time.time()

        # now I have to compute the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        # now I have also to compute the Hvp between hessian and velocity
        start = time.time()
        if batchify:
            hvp_data_fitting = 0
            for batch_img, batch_label in self.X:
                if self.type == "fc":
                    _, result = custum_hvp(
                        self.CE_loss,
                        (params, (batch_img, batch_label)),
                        (vel_as_params, (torch.zeros_like(batch_img), torch.zeros_like(batch_label))),
                    )
                else:
                    print("If you are getting an error, before here I was using self.CE_loss2, so double check that")
                    _, result = custum_hvp(
                        self.CE_loss,
                        (params, (batch_img, batch_label)),
                        (vel_as_params, (torch.zeros_like(batch_img), torch.zeros_like(batch_label))),
                    )
                hvp_data_fitting += torch.cat([sub_prod.flatten() for sub_prod in result])
        else:
            _, result = custum_hvp(
                self.CE_loss, (params, data), (vel_as_params, (torch.zeros_like(data[0]), torch.zeros_like(data[1])))
            )
            hvp = [sub_prod.flatten() for sub_prod in result]
            hvp_data_fitting = torch.cat(hvp)

        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity
            tot_hvp = hvp_data_fitting + hvp_reg.view(-1)
        else:
            tot_hvp = hvp_data_fitting

        tot_hvp = tot_hvp.to(self.device)
        tot_gradient = tot_gradient.to(self.device)
        end = time.time()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.view(-1, 1).detach().cpu().numpy(), tot_hvp.view(-1, 1).detach().cpu().numpy()
        else:
            return second_derivative.view(-1, 1).detach().cpu().numpy()

    def get_gradient_value_in_specific_point(self, current_point):
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # method to return the gradient of the loss in a specific point
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float()

        current_point = current_point.to(self.device)
        assert (
            len(current_point) == self.n_params
        ), "You are passing a larger vector than the number of weights in the model"
        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        grad_data_fitting_term = 0
        if self.batching:
            for batch_img, batch_label in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label))
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            data = (self.X, self.y)
            grad_per_example = self.compute_grad_data_fitting_term(params, data)

            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)

        # now I have to compute the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)
        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg
        tot_gradient = tot_gradient.to(self.device)

        return tot_gradient


class linearized_cross_entropy_manifold:
    """
    Also in this case I have to separate data fitting term and regularization term for gradient and
    hessian computation in case of batches.
    """

    def __init__(
        self,
        model,
        X,
        y,
        f_MAP,
        theta_MAP,
        batching=False,
        device="cpu",
        lambda_reg=None,
        type="fc",
        N=None,
        B1=None,
        B2=None,
    ):
        self.model = model
        # TODO: decide if it is better to pass X and Y or
        # pass data that is either data = (X,y) or a dataloader
        self.X = X
        self.y = y
        self.N = len(self.X)
        self.n_params = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        self.batching = batching
        self.device = device
        self.type = type
        # self.prior_precision = prior_precision
        self.lambda_reg = lambda_reg
        assert y is None if batching else True, "If batching is True, y should be None"

        self.theta_MAP = theta_MAP
        self.f_MAP = f_MAP

        # these are just to avoid to have to pass them as input of the functions
        # we use to compute the gradient and the hessian vector product
        self.fmodel = None
        self.buffers = None

        self.fmodel_map = None
        self.params_map = None
        self.buffers_map = None

        self.N = N
        self.B1 = B1
        self.B2 = B2
        self.factor = None

        if self.B1 is not None:
            self.factor = N / B1
            if self.B2 is not None:
                self.factor = self.factor * (B2 / B1)

    @staticmethod
    def is_diagonal():
        return False

    def CE_loss(self, param, data, f_MAP):
        """
        Data fitting term of the loss
        """

        def predict(params, datas):
            y_preds = self.fmodel_map(params, self.buffers_map, datas)
            return y_preds

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        if self.type != "fc":
            x = x.unsqueeze(1)

        params_map = get_params_structure(self.theta_MAP, param)
        diff_weights = []
        for i in range(len(param)):
            diff_weights.append(param[i] - self.params_map[i])
        diff_weights = tuple(diff_weights)
        _, jvp_value = jvp(predict, (params_map, x), (diff_weights, torch.zeros_like(x)), strict=False)

        y_pred = f_MAP + jvp_value

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if self.type == "fc":
            if self.factor is not None:
                return self.factor * criterion(y_pred, y)
            else:
                return criterion(y_pred, y)
        else:
            if self.factor is not None:
                return self.factor * criterion(y_pred.view(-1), y)
            else:
                return criterion(y_pred.view(-1), y)

    def L2_norm(self, param):
        """
        L2 regularization. I need this separate from the loss for the gradient computation
        """
        w_norm = sum([sum(w.view(-1) ** 2) for w in param])
        return self.lambda_reg * w_norm

    def compute_grad_data_fitting_term(self, params, data, f_MAP):
        # TODO: understand how to make vmap work without passing the data
        ft_compute_grad = grad(self.CE_loss)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params, data, f_MAP)
        return ft_per_sample_grads

    def compute_grad_L2_reg(self, params):
        ft_compute_grad = grad(self.L2_norm)
        # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0))
        # the input of this function is just the parameters, because
        # we are accessing the data from the class
        ft_per_sample_grads = ft_compute_grad(params)
        return ft_per_sample_grads

    def geodesic_system(self, current_point, velocity, return_hvp=False):
        # check if they are numpy array or torch.Tensor
        # here we need torch tensor to perform these operations
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float().to(self.device)

        if not isinstance(velocity, torch.Tensor):
            velocity = torch.from_numpy(velocity).float().to(self.device)

        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # let's start by putting the current points into the model
        torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())

        # now I have everything to compute the the second derivative
        # let's compute the gradient
        start = time.time()
        grad_data_fitting_term = 0
        if batchify:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label), batch_MAP)
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            grad_per_example = self.compute_grad_data_fitting_term(params, data, self.f_MAP)
            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)
        end = time.time()

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg

        vel_as_params = get_params_structure(velocity, params)

        start = time.time()
        hvp_data_fitting = 0
        if batchify:
            for batch_img, batch_label, batch_f_MAP in self.X:
                if self.type == "fc":
                    _, result = custum_hvp(
                        self.CE_loss,
                        (params, (batch_img, batch_label), batch_f_MAP),
                        (
                            vel_as_params,
                            (torch.zeros_like(batch_img), torch.zeros_like(batch_label)),
                            torch.zeros_like(batch_f_MAP),
                        ),
                    )
                else:
                    _, result = custum_hvp(
                        self.CE_loss,
                        (params, (batch_img, batch_label), batch_f_MAP),
                        (
                            vel_as_params,
                            (torch.zeros_like(batch_img), torch.zeros_like(batch_label)),
                            torch.zeros_like(batch_f_MAP),
                        ),
                    )

                hvp_data_fitting += torch.cat([sub_prod.flatten() for sub_prod in result])
        else:
            _, result = custum_hvp(
                self.CE_loss,
                (params, data, self.f_MAP),
                (vel_as_params, (torch.zeros_like(data[0]), torch.zeros_like(data[1])), torch.zeros_like(self.f_MAP)),
            )

            hvp = [sub_prod.flatten() for sub_prod in result]
            hvp_data_fitting = torch.cat(hvp)

        # I have to add the hvp of the regularization term
        if self.lambda_reg is not None:
            hvp_reg = 2 * self.lambda_reg * velocity

            tot_hvp = hvp_data_fitting + hvp_reg.view(-1)
        else:
            tot_hvp = hvp_data_fitting

        tot_hvp = tot_hvp.to(self.device)
        tot_gradient = tot_gradient.to(self.device)
        end = time.time()

        second_derivative = -((tot_gradient / (1 + tot_gradient.T @ tot_gradient)) * (velocity.T @ tot_hvp)).flatten()

        if return_hvp:
            return second_derivative.view(-1, 1).detach().cpu().numpy(), tot_hvp.view(-1, 1).detach().cpu().numpy()
        else:
            return second_derivative.view(-1, 1).detach().cpu().numpy()

    def get_gradient_value_in_specific_point(self, current_point):
        if isinstance(self.X, torch.utils.data.DataLoader):
            batchify = True
        else:
            batchify = False
            data = (self.X, self.y)

        # method to return the gradient of the loss in a specific point
        if not isinstance(current_point, torch.Tensor):
            current_point = torch.from_numpy(current_point).float()

        current_point = current_point.to(self.device)
        assert (
            len(current_point) == self.n_params
        ), "You are passing a larger vector than the number of weights in the model"
        self.model.zero_grad()

        self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

        grad_data_fitting_term = 0
        if batchify:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            for batch_img, batch_label, batch_MAP in self.X:
                grad_per_example = self.compute_grad_data_fitting_term(params, (batch_img, batch_label), batch_MAP)
                grad_data_fitting_term += stack_gradient3(grad_per_example, self.n_params).view(-1, 1)
        else:
            self.model.zero_grad()
            torch.nn.utils.vector_to_parameters(current_point, self.model.parameters())
            self.fmodel, params, self.buffers = make_functional_with_buffers(self.model)

            torch.nn.utils.vector_to_parameters(self.theta_MAP, self.model.parameters())
            self.fmodel_map, self.params_map, self.buffers_map = make_functional_with_buffers(self.model)

            grad_per_example = self.compute_grad_data_fitting_term(params, data, self.f_MAP)
            gradw = stack_gradient3(grad_per_example, self.n_params)
            grad_data_fitting_term = gradw.view(-1, 1)
        end = time.time()

        # here now I have to compute also the gradient of the regularization term
        if self.lambda_reg is not None:
            # I have to compute the L2 reg gradient
            grad_reg = self.compute_grad_L2_reg(params)
            grad_reg = stack_gradient3(grad_reg, self.n_params)
            grad_reg = grad_reg.view(-1, 1)

        else:
            grad_reg = 0

        tot_gradient = grad_data_fitting_term + grad_reg
        tot_gradient = tot_gradient.to(self.device)

        return tot_gradient
