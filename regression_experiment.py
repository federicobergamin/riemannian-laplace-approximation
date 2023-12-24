"""
TODO:
1) CASE WHERE WE NOT OPTIMIZE THE PRIOR BUT WE OPTIMIZE THE NOISE VAR
"""

from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# sys.path.append("laplace")
sys.path.append("stochman")
import matplotlib
import matplotlib.colors as colors
from laplace import Laplace
import seaborn as sns
import geomai.utils.geometry as geometry
from torch.distributions import MultivariateNormal

# NOTE: I ma using an old version of functorch:0.2.1 instead of the new one
## because the new one breaks backpack libray and also it does not support torchvision
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap
from functools import partial
from functorch_utils import get_params_structure, stack_gradient, custum_hvp, stack_gradient2
from torch.distributions import Normal

from manifold import regression_manifold
from sklearn.model_selection import train_test_split
from utils.metrics import accuracy, nll, brier, calibration, nlpd_using_predictions
from tqdm import tqdm
import argparse


def parallel_expmap(item, V, manifold, w_map):
    v = V[item, :].reshape(-1, 1)
    curve, failed = geometry.expmap(manifold, w_map.clone().numpy(), v)
    geo_weights = curve(1)[0]  # getting intermediate points in the curve (<1.0), we can get more weights
    return geo_weights


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    font = {"family": "serif", "size": 12}
    matplotlib.rc("font", **font)

    subset_of_weights = args.subset  # either 'last_layer' or 'all'
    hessian_structure = args.structure
    if subset_of_weights == "last_layer":
        n_posterior_samples = args.samples
    else:
        n_posterior_samples = args.samples
    optimize_also_noise_var = args.optimize_sigma
    print("Hessian structure: ", hessian_structure)
    print("Subset of weights: ", subset_of_weights)

    assert hessian_structure == "full" or hessian_structure == "diag", "Hessian structure not supported"
    assert subset_of_weights == "all" or subset_of_weights == "last_layer", "subset_of_weights not supported"

    Xtrain = np.loadtxt("data/snelson_data/train_inputs")
    Ytrain = np.loadtxt("data/snelson_data/train_outputs")
    Xtest = np.loadtxt("data/snelson_data/test_inputs")

    # # NOTE: IN THIS CASE WE ARE ALSO SPLITTING THE TRAINING SET INTO TRAIN AND TEST WITH DIFFERENT SEEDS
    # Xtrain, Xtest_in_region, Ytrain, Ytest_in_region = train_test_split(Xtrain, Ytrain, test_size=0.25, random_state=230)

    if args.in_between:
        # here I have to create the in-between uncertainty dataset
        ind_in_between = np.where(np.logical_and(Xtrain >= 1.5, Xtrain <= 3))[0]
        ind_not_in_between = np.where(np.logical_or(Xtrain < 1.5, Xtrain > 3))[0]

        new_Xtrain = Xtrain[ind_not_in_between]
        new_Ytrain = Ytrain[ind_not_in_between]

        Xtest_in_region = Xtrain[ind_in_between]
        Ytest_in_region = Ytrain[ind_in_between]

        Xtrain = new_Xtrain
        Ytrain = new_Ytrain

    else:
        # NOTE: IN THIS CASE WE ARE ALSO SPLITTING THE TRAINING SET INTO TRAIN AND TEST WITH DIFFERENT SEEDS
        Xtrain, Xtest_in_region, Ytrain, Ytest_in_region = train_test_split(
            Xtrain, Ytrain, test_size=0.25, random_state=230
        )

        print("N of observations")
        print(Xtrain.shape)

    Xtrain = torch.from_numpy(Xtrain).float().view(-1, 1)
    Ytrain = torch.from_numpy(Ytrain).float()
    print("Var ytrain: ", torch.var(Ytrain))
    Xtest_in_region = torch.from_numpy(Xtest_in_region).float().view(-1, 1)
    Ytest_in_region = torch.from_numpy(Ytest_in_region).float()

    if args.batch_data:
        dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain.view(-1, 1))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)

    if args.dropout:
        exp = "dropout"
    elif args.small_model:
        exp = "small"
    else:
        exp = "normal"

    # Network definition
    if exp == "normal":
        H = 10  # hidden layer size ----> # maybe add rejection step for H>6
        # activ_fun = torch.nn.Tanh()  # With ReLU I am getting some strange Hessian
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh(), torch.nn.Linear(H, 1)
        )

        if args.optimizer == "adam":
            print("USING Adam")
            weight_decay = 1e-2
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=weight_decay)
            max_epoch = 500  # 2000#100000#80000#80000#1000#40000 #80000 # 1500
        else:
            print("USING SGD")
            weight_decay = 1e-2
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            max_epoch = 35000  # 700000#35000

    elif exp == "dropout":
        print("Using model with dropout")
        # model used with dropout
        H = 10
        dropout_prob = 0.01
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(H, H),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(H, 1),
        )

        if args.optimizer == "adam":
            print("USING Adam")
            weight_decay = 1e-4
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            max_epoch = 40000  # 40000#120000#80000#80000#1000#40000 #80000 # 1500
        else:
            print("USING SGD")
            weight_decay = 1e-4
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            max_epoch = 950000  # 35000

    else:
        # we are using the small model
        print("Using smallest model")
        H = 15
        model = torch.nn.Sequential(torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, 1))

        if args.optimizer == "adam":
            print("USING Adam")
            weight_decay = 1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            max_epoch = 20000  # 80000#80000#1000#40000 #80000 # 1500
        else:
            print("USING SGD")
            weight_decay = 1e-3  # 1e-6
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)
            max_epoch = 700000  # 35000

    # Adam settings

    # weight_decay = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    # max_epoch = 100000#80000#80000#1000#40000 #80000 # 1500

    # weight_decay = 1e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    # max_epoch = 700000#35000

    print(model)
    # sigma = 0.01# this is our regression model
    # I can train it
    # Train the network
    # max_epoch = 1000
    if args.batch_data:
        i = 0
        for epoch in range(max_epoch):
            for batch_x, batch_y in train_loader:
                i += 1
                y_pred = model(batch_x)
                loss = torch.sum((y_pred.view(-1, 1) - batch_y) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    else:
        for epoch in range(max_epoch):
            y_pred = model(Xtrain)
            loss = 0.5 * torch.sum((y_pred.view(-1) - Ytrain) ** 2)  # / (2*(sigma**2))
            # print(torch.sum((y_pred - Y_train)**2) / 2*0.1**2)
            # print(torch.sum((y_pred/np.sqrt(2)*0.1 - Y_train/np.sqrt(2)*0.1)**2))
            # print(loss)
            if epoch % 10000 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    Xtest = torch.from_numpy(Xtest).float().view(-1, 1)
    # print(Xtest)
    model.eval()
    with torch.no_grad():
        y_pred_test = model(Xtest)

    plt.scatter(Xtrain, Ytrain)
    plt.plot(Xtest, y_pred_test)
    plt.show()

    # save the MAP estimate
    map_solution = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    print("Check map solution")
    print(map_solution[0:10])
    print(map_solution.shape)

    if not args.batch_data:
        dataset = torch.utils.data.TensorDataset(Xtrain, Ytrain.view(-1, 1))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(Xtrain), shuffle=True)

    if subset_of_weights == "last_layer":
        la = Laplace(
            model,
            "regression",
            subset_of_weights="last_layer",
            hessian_structure=hessian_structure,
            prior_precision=2 * weight_decay,
        )
        la.fit(train_loader)
        print("args.optimize_prior last layer: ", args.optimize_prior)

        if args.optimize_prior:
            if optimize_also_noise_var:
                log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
                hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-2)
                for i in range(3000):
                    hyper_optimizer.zero_grad()
                    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                    neg_marglik.backward()
                    hyper_optimizer.step()
            else:
                la.optimize_prior_precision(method="marglik")
        else:
            if optimize_also_noise_var:
                log_prior, log_sigma = torch.log(torch.tensor([weight_decay])), torch.ones(1, requires_grad=True)
                hyper_optimizer = torch.optim.Adam([log_sigma], lr=1e-2)
                for i in range(3000):
                    hyper_optimizer.zero_grad()
                    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                    neg_marglik.backward()
                    hyper_optimizer.step()
    else:
        if args.optimize_prior:
            jitter_laplace = 0.0
        else:
            jitter_laplace = 0.001
        print(f"I am using a jitter for Laplace {jitter_laplace}")
        la = Laplace(
            model,
            "regression",
            subset_of_weights="all",
            hessian_structure=hessian_structure,
            prior_precision=2 * weight_decay + jitter_laplace,
        )
        la.fit(train_loader)
        print("args.optimize_prior full: ", args.optimize_prior)
        if args.optimize_prior:
            if hessian_structure == "diag":
                if optimize_also_noise_var:
                    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
                    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
                    for i in range(3000):
                        hyper_optimizer.zero_grad()
                        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                        neg_marglik.backward()
                        hyper_optimizer.step()
                else:
                    la.optimize_prior_precision(method="marglik")
            else:
                if optimize_also_noise_var:
                    # here it means that we are full full. I guess we can set the noise to 0.5 and optimize only the noise variance
                    # log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.zeros(1, requires_grad=False)
                    log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
                    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-3)
                    for i in range(3000):
                        hyper_optimizer.zero_grad()
                        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                        neg_marglik.backward()
                        hyper_optimizer.step()
                else:
                    la.optimize_prior_precision(method="marglik")
        else:
            if optimize_also_noise_var:
                log_prior, log_sigma = torch.log(torch.tensor([2 * weight_decay + jitter_laplace])), torch.ones(
                    1, requires_grad=True
                )
                hyper_optimizer = torch.optim.Adam([log_sigma], lr=1e-1)
                for i in range(3000):
                    hyper_optimizer.zero_grad()
                    neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
                    neg_marglik.backward()
                    hyper_optimizer.step()

    print("Learned parameters")
    print("prior precision: ", la.prior_precision)
    print("sigma: ", la.sigma_noise)
    print("H factor: ", la._H_factor)

    # now I have to create the first manifold just to get the Hessian in our case
    # I start by sampling the initial velocities for the classical Laplace

    ###### CLASSIC LAPLACE SAMPLES ########
    if subset_of_weights == "last_layer":
        if hessian_structure == "full":
            dist = MultivariateNormal(loc=torch.zeros(H + 1), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,))
            V_LA = V_LA.detach().numpy()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().numpy()
    else:
        if hessian_structure == "full":
            la2 = Laplace(
                model,
                "regression",
                subset_of_weights="all",
                hessian_structure=hessian_structure,
                prior_precision=la.prior_precision.item(),
            )
            la2.sigma_noise = la.sigma_noise / 10
            # la2._H_factor = 100 * (1/la.sigma_noise**2)
            print("H factor: ", la2._H_factor)
            dist = MultivariateNormal(loc=torch.zeros_like(map_solution), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,))
            V_LA = V_LA.detach().numpy()
        else:
            print("son qua giusto?")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().numpy()

    ######## OUR LAPLACE APPROXIMATION #######
    # TODO: maybe this part can be implemented in a nicer way, since I am defining the
    # manifold twice, and that is not elegant tbh

    if exp == "normal":
        model2 = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.Tanh(),
            # torch.nn.Dropout(p=0.01),
            torch.nn.Linear(H, H),
            torch.nn.Tanh(),
            # torch.nn.Dropout(p=0.01),
            torch.nn.Linear(H, 1),
        )
    elif exp == "dropout":
        model2 = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(H, H),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(H, 1),
        )
    else:
        model2 = torch.nn.Sequential(torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, 1))

    torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())

    model2.eval()
    if subset_of_weights == "last_layer":
        # here i have to split the weights and the model
        # or at least being able to use only the last layer weights
        # TODO: here I am using H but it should be the number of weights of the layer before
        # the final one
        n_last_layer_weights = H + 1
        # r_MAP = None
        # here I have also to precompute some stuff for the last layer
        # I have to compute the feature vector R before the last layer
        MAP = map_solution.clone()
        feature_extractor_map = MAP[0 : len(MAP) - n_last_layer_weights].clone()
        print(feature_extractor_map.shape)
        ll_map = MAP[-n_last_layer_weights:].clone()
        print(ll_map.shape)
        # now to define the two different models

        if exp == "normal":
            feature_extractor_model = torch.nn.Sequential(
                torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh()
            )
            ll = torch.nn.Linear(H, 1)
        elif exp == "dropout":
            feature_extractor_model = torch.nn.Sequential(
                torch.nn.Linear(1, H), torch.nn.Tanh(), torch.nn.Linear(H, H), torch.nn.Tanh()
            )
            ll = torch.nn.Linear(H, 1)
        else:
            feature_extractor_model = torch.nn.Sequential(
                torch.nn.Linear(1, H),
                # torch.nn.Tanh(),
                # torch.nn.Linear(H, H),
                torch.nn.Tanh(),
            )
            ll = torch.nn.Linear(H, 1)

        # and use the correct weights
        torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
        torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

        # I have to compute the last layer activations
        with torch.no_grad():
            if args.batch_data:
                R = torch.zeros((len(Xtrain), H))
                idx = 0
                for batch_img, _ in train_loader:
                    R[idx : min(idx + len(batch_img), len(Xtrain)), :] = feature_extractor_model(batch_img)
                    idx += len(batch_img)
                # R = feature_extractor_model(Xtrain)
            else:
                R = feature_extractor_model(Xtrain)
            print("R shape ", R.shape)
        # and now I can create the manifold
        if args.batch_data:
            raise NotImplementedError("Batching not supported")
            if args.optimize_prior and optimize_also_noise_var:
                manifold = regression_manifold(
                    ll, Xtrain, Ytrain.view(-1, 1), prior_precision=la.prior_precision.item(), noise_var=1.0
                )
        else:
            # ma
            # manifold = last_layer_regression_manifold(name=None, R=R,  NOF=len(Xtrain), Y=Ytrain.view(-1,1))
            if args.optimize_prior and optimize_also_noise_var:
                print("last layer, opt prior and noise")
                manifold = regression_manifold(
                    ll,
                    R,
                    Ytrain.view(-1, 1),
                    lambda_reg=la.prior_precision.item() / 2,
                    noise_var=la.sigma_noise.item() ** 2,
                )
            elif args.optimize_prior and not optimize_also_noise_var:
                print("last layer, opt prior and not noise")
                manifold = regression_manifold(
                    ll, R, Ytrain.view(-1, 1), lambda_reg=la.prior_precision.item() / 2, noise_var=1.0
                )
            elif not args.optimize_prior and optimize_also_noise_var:
                print("last layer, not opt prior but noise opt")
                manifold = regression_manifold(
                    ll, R, Ytrain.view(-1, 1), lambda_reg=weight_decay, noise_var=la.sigma_noise.item() ** 2
                )
            else:
                print("last layer, not opt prior and noise")
                # we do not optimize anything
                manifold = regression_manifold(ll, R, Ytrain.view(-1, 1), lambda_reg=weight_decay, noise_var=1.0)
    else:
        if args.batch_data:
            manifold = regression_manifold(model2, train_loader, None, batching=True)
        else:
            if args.optimize_prior and optimize_also_noise_var:
                print("optimize both")
                manifold = regression_manifold(
                    model2,
                    Xtrain,
                    Ytrain.view(-1, 1),
                    lambda_reg=la.prior_precision.item() / 2,
                    noise_var=la.sigma_noise.item() ** 2,
                )
            elif args.optimize_prior and not optimize_also_noise_var:
                manifold = regression_manifold(
                    model2, Xtrain, Ytrain.view(-1, 1), lambda_reg=la.prior_precision.item() / 2, noise_var=2.0
                )
            elif not args.optimize_prior and optimize_also_noise_var:
                print("no prior optim but noise")
                manifold = regression_manifold(
                    model2, Xtrain, Ytrain.view(-1, 1), lambda_reg=weight_decay, noise_var=la.sigma_noise.item() ** 2
                )
            else:
                manifold = regression_manifold(
                    model2, Xtrain, Ytrain.view(-1, 1), lambda_reg=weight_decay, noise_var=1.0
                )

    ## all ingredients for computing the classic Laplace weights
    # I can compute the weights we get from classic Laplace
    ### I can compute directly the weights used by Laplace
    if subset_of_weights == "last_layer":
        weights_LA = np.zeros((n_posterior_samples, len(map_solution)))
        for i in range(n_posterior_samples):
            laplace_weigths = torch.from_numpy(V_LA[i, :].reshape(-1)).float().to(device) + ll_map.clone()
            laplace_weigths = torch.cat((feature_extractor_map.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[i, :] = laplace_weigths.cpu().numpy()
    else:
        weights_LA = np.zeros((n_posterior_samples, len(map_solution)))
        for i in range(n_posterior_samples):
            laplace_weigths = torch.from_numpy(V_LA[i, :].reshape(-1)).float().to(device) + map_solution
            # laplace_weigths = torch.cat((feature_extractor_MAP.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[i, :] = laplace_weigths.cpu().numpy()

    # now I have to compute the A matrix using Søren's approximation
    # with torch.no_grad():
    #     if subset_of_weights == "last_layer":
    #         grad = manifold.get_gradient_value_in_specific_point(ll_map.clone())
    #     else:
    #         grad = manifold.get_gradient_value_in_specific_point(map_solution.clone())

    # # now I can compute the A
    # if subset_of_weights == "last_layer":
    #     # NOTE: if I use a jitter for Laplace then I am getting a biased Hessian here
    #     H_factor = 1 / la.sigma_noise**2
    #     Hess = la._H_factor * la.H + (la.prior_precision * torch.eye(len(ll_map)).to(device))

    #     with torch.no_grad():
    #         gradTgrad = grad.T @ grad
    #         small_a_numerator = -1.0 + 1.0 / (torch.sqrt(1.0 + gradTgrad))
    #         small_a = small_a_numerator / gradTgrad

    #         A = torch.eye(len(ll_map)).to(device) + small_a * grad * grad.T
    #         print(A.shape)

    #     print(A)

    #     new_Hess = A.T @ Hess @ A
    #     symm_new_Hessian = (new_Hess + new_Hess.T) * 0.5
    #     solved = False
    #     jitters = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    #     idx_jitter = 0
    #     while not solved:
    #         if idx_jitter > len(jitters):
    #             raise ValueError("You should increase value of jitter")
    #         try:
    #             new_cov = get_inverse(symm_new_Hessian + torch.eye(len(ll_map)).to(device) * jitters[idx_jitter])
    #             print(new_cov[0:5, 0:5])
    #             # solved_inversion = True
    #             new_dist = MultivariateNormal(
    #                 loc=torch.zeros(n_last_layer_weights).to(device), covariance_matrix=new_cov
    #             )
    #             solved = True
    #             print(f"The jitter we used in this case is {jitters[idx_jitter]}")
    #         except (torch._C._LinAlgError, ValueError):
    #             # print('We are pretty unlucky, we finished all the jitters mate')
    #             idx_jitter += 1

    #     V_normal_coord = new_dist.sample((n_posterior_samples,))
    #     V_tangent_coord = V_normal_coord @ A.T
    #     V_tangent_coord = (
    #         V_tangent_coord.detach().cpu().numpy()
    #     )  # (V_tangent_coord * torch.sqrt(torch.linalg.det(torch.eye(len(map_solution))+ grad * grad.T))).detach().cpu().numpy()
    # else:
    #     # print("Refitting Laplace for our method now, don't care if what I get is not invertible because I am changing everything")
    #     # jitter_laplace = 0.0
    #     # print(f'I am using a jitter for Laplace {jitter_laplace}')
    #     # la = Laplace(model, 'regression',
    #     #             subset_of_weights='all',
    #     #             hessian_structure=hessian_structure,
    #     #             prior_precision = 2*weight_decay + jitter_laplace)
    #     # la.fit(train_loader)

    #     # if args.optimize_prior:
    #     #     if hessian_structure == 'diag':
    #     #         if optimize_also_noise_var:
    #     #             log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    #     #             hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    #     #             for i in range(3000):
    #     #                 hyper_optimizer.zero_grad()
    #     #                 neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    #     #                 neg_marglik.backward()
    #     #                 hyper_optimizer.step()
    #     #         else:
    #     #             la.optimize_prior_precision(method='marglik')
    #     #     else:
    #     #         if optimize_also_noise_var:
    #     #             # here it means that we are full full. I guess we can set the noise to 0.5 and optimize only the noise variance
    #     #             # log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.zeros(1, requires_grad=False)
    #     #             log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
    #     #             hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-3)
    #     #             for i in range(3000):
    #     #                 hyper_optimizer.zero_grad()
    #     #                 neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    #     #                 neg_marglik.backward()
    #     #                 hyper_optimizer.step()
    #     #         else:
    #     #             la.optimize_prior_precision(method='marglik')
    #     # else:
    #     #     if optimize_also_noise_var:
    #     #         log_prior, log_sigma = torch.log(torch.tensor([2*weight_decay+jitter_laplace])), torch.ones(1, requires_grad=True)
    #     #         hyper_optimizer = torch.optim.Adam([log_sigma], lr=1e-1)
    #     #         for i in range(3000):
    #     #             hyper_optimizer.zero_grad()
    #     #             neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    #     #             neg_marglik.backward()
    #     #             hyper_optimizer.step()

    #     H_factor = 1 / la.sigma_noise**2
    #     if args.structure == "diag":
    #         Hess = H_factor * torch.diag(la.H) + (la.prior_precision * torch.eye(len(map_solution)).to(device))
    #     else:
    #         Hess = H_factor * la.H + (la.prior_precision * torch.eye(len(map_solution)).to(device))

    #     with torch.no_grad():
    #         gradTgrad = grad.T @ grad
    #         small_a_numerator = -1.0 + 1.0 / (torch.sqrt(1.0 + gradTgrad))
    #         small_a = small_a_numerator / gradTgrad

    #         A = torch.eye(len(map_solution)).to(device) + small_a * grad * grad.T
    #         print(A.shape)

    #     new_Hess = A.T @ Hess @ A
    #     symm_new_Hessian = (new_Hess + new_Hess.T) * 0.5
    #     solved = False
    #     jitters = np.linspace(
    #         0, 0.5, 1000
    #     )  # [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    #     idx_jitter = 0
    #     while not solved:
    #         if idx_jitter > len(jitters) - 1:
    #             raise ValueError("You should increase value of jitter")
    #         try:
    #             print(idx_jitter)
    #             new_cov = get_inverse(symm_new_Hessian + torch.eye(len(map_solution)).to(device) * jitters[idx_jitter])
    #             new_dist = MultivariateNormal(loc=torch.zeros(len(map_solution)).to(device), covariance_matrix=new_cov)
    #             solved = True
    #             print(f"The jitter we used in this case is {jitters[idx_jitter]}")
    #         except (torch._C._LinAlgError, ValueError):
    #             idx_jitter += 1

    #     V_normal_coord = new_dist.sample((n_posterior_samples,))
    #     V_tangent_coord = V_normal_coord @ A.T

    #     V_tangent_coord = V_tangent_coord

    # now for security reasons I'll re-initialize the manifold
    if subset_of_weights == "last_layer":
        torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
        torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())
        if args.optimize_prior and optimize_also_noise_var:
            manifold = regression_manifold(
                ll,
                R,
                Ytrain.view(-1, 1),
                lambda_reg=la.prior_precision.item() / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )
        elif args.optimize_prior and not optimize_also_noise_var:
            manifold = regression_manifold(
                ll,
                R,
                Ytrain.view(-1, 1),
                lambda_reg=la.prior_precision.item() / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )
        elif not args.optimize_prior and optimize_also_noise_var:
            manifold = regression_manifold(
                ll,
                R,
                Ytrain.view(-1, 1),
                lambda_reg=weight_decay / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )
        else:
            # we do not optimize anything
            manifold = regression_manifold(
                ll,
                R,
                Ytrain.view(-1, 1),
                lambda_reg=weight_decay / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )
    else:
        torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
        if args.optimize_prior and optimize_also_noise_var:
            manifold = regression_manifold(
                model2,
                Xtrain,
                Ytrain.view(-1, 1),
                lambda_reg=la.prior_precision.item() / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )
        elif args.optimize_prior and not optimize_also_noise_var:
            manifold = regression_manifold(
                model2, Xtrain, Ytrain.view(-1, 1), lambda_reg=la.prior_precision.item() / 2, noise_var=1.0
            )
        elif not args.optimize_prior and optimize_also_noise_var:
            manifold = regression_manifold(
                model2,
                Xtrain,
                Ytrain.view(-1, 1),
                lambda_reg=weight_decay / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )
        else:
            manifold = regression_manifold(
                model2,
                Xtrain,
                Ytrain.view(-1, 1),
                lambda_reg=weight_decay / 2,
                noise_var=la.sigma_noise.item() ** 2,
            )

    weights_ours = torch.zeros(n_posterior_samples, len(map_solution))

    # now I can compute my weights
    for n in tqdm(range(n_posterior_samples), desc="expmapping init velocities"):
        v = torch.from_numpy(V_LA[n, :]).float().reshape(-1, 1)
        if subset_of_weights == "last_layer":
            curve, failed = geometry.expmap(manifold, ll_map.clone(), v)
            Wb_sample = curve(1)[0]
            _new_weights = torch.cat(
                (feature_extractor_map.view(-1), torch.from_numpy(Wb_sample).float().view(-1)), dim=0
            )
            weights_ours[n, :] = _new_weights.view(-1)
        else:
            curve, failed = geometry.expmap(manifold, map_solution, v)
            _new_weights = curve(1)[0]
            weights_ours[n, :] = torch.from_numpy(_new_weights.reshape(-1))

    # now I can evaluate the weigths
    pred_ours = torch.zeros((n_posterior_samples, len(Xtest)))
    pred_laplace = torch.zeros((n_posterior_samples, len(Xtest)))
    prediction_in_region_LA = torch.zeros((n_posterior_samples, len(Xtest_in_region)))
    prediction_in_region_ours = torch.zeros((n_posterior_samples, len(Xtest_in_region)))

    # start with Laplace prediction
    for n in range(n_posterior_samples):
        w_LA = torch.from_numpy(weights_LA[n, :]).float()
        torch.nn.utils.vector_to_parameters(w_LA, model2.parameters())
        with torch.no_grad():
            y_pred_test = model2(Xtest)
            pred_laplace[n, :] = y_pred_test.view(-1)

            # now I have also to predict the in-data-region points
            y_pred_in_region = model2(Xtest_in_region)
            prediction_in_region_LA[n, :] = y_pred_in_region.view(-1)
        plt.plot(Xtest, y_pred_test, linewidth=1, c="red", alpha=0.2)
        if n == 1:
            plt.plot(Xtest, y_pred_test, linewidth=1, c="red", alpha=0.2, label="LA")

    for n in range(n_posterior_samples):
        w_our = weights_ours[n, :]
        torch.nn.utils.vector_to_parameters(w_our, model2.parameters())
        with torch.no_grad():
            y_pred_test = model2(Xtest)
            pred_ours[n, :] = y_pred_test.view(-1)

            # now I have also to predict the in-data-region points
            y_pred_in_region = model2(Xtest_in_region)
            prediction_in_region_ours[n, :] = y_pred_in_region.view(-1)

        plt.plot(Xtest, y_pred_test, linewidth=1, c="blue", alpha=0.2)
        if n == 1:
            plt.plot(Xtest, y_pred_test, linewidth=1, c="blue", alpha=0.2, label="Ours")

    torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
    with torch.no_grad():
        map_pred = model2(Xtest)
    plt.scatter(Xtrain, Ytrain, c="black", zorder=2, alpha=0.5)
    plt.plot(Xtest, map_pred, linewidth=2, c="green", alpha=1.0, label="MAP", zorder=2)

    pred_ours_mean = pred_ours[0:n_posterior_samples, :].mean(0)
    pred_laplace_mean = pred_laplace[0:n_posterior_samples, :].mean(0)
    plt.plot(Xtest, pred_laplace_mean, linewidth=2, c="yellow", alpha=1.0, label="Laplace mean", zorder=2)
    plt.plot(Xtest, pred_ours_mean, linewidth=2, c="pink", alpha=1.0, label="Our mean", zorder=2)
    # plt.xlim([min(Xtest), max(Xtest)])
    plt.xlim([-0.5, 6.5])
    plt.ylim([-4, 4])
    plt.legend(loc=2)
    # plt.title('Posterior samples')
    # plt.savefig("plot_regression_rebuttal/posterior.pdf")
    plt.show()

    # I have to save everyhting I need for the regression data
    # DIR = "plots_regression_data_in_between/"
    # torch.save(Xtest, DIR + "Xtest.pt")
    # torch.save(Xtrain, DIR + "Xtrain.pt")
    # torch.save(Ytrain, DIR + "Ytrain.pt")
    # torch.save(Xtest_in_region, DIR + "Xtest_in_region.pt")
    # torch.save(Ytest_in_region, DIR + "Ytest_in_region.pt")
    # torch.save(la.sigma_noise.item(), DIR + "sigma_noise_big_model.py")
    # torch.save(pred_ours, DIR + "our_predictions_bigmodel.pt")
    # torch.save(pred_laplace, DIR + "laplace_pred_big_model.pt")
    # torch.save(pred_laplace_mean, DIR + "laplace_mean_big_model.pt")
    # torch.save(pred_ours_mean, DIR + "our_mean_big_model.pt")
    # torch.save(map_pred, DIR + "map_pred_big_model.pt")

    ## now I should also create the plot of the final predictive and compute the NLPD for the test set
    # at this point I can compute the predictuive distribution for the plot, just to
    ## compare the two approaches, and maybe I can also plot the linearization on top
    ## to compare the three things

    f_mean_LA = pred_laplace_mean.detach().numpy()
    f_mean_OUR = pred_ours_mean.detach().numpy()

    # I can compute the variance of the prediction
    f_var_LA = pred_laplace.var(0).detach()
    f_var_OUR = pred_ours.var(0).detach()

    std_pred_LA = torch.sqrt(f_var_LA + la.sigma_noise.square()).detach().numpy()
    std_pred_OUR = torch.sqrt(f_var_OUR + la.sigma_noise.square()).detach().numpy()

    print(Xtest.shape)
    print(std_pred_LA.shape)
    print(std_pred_OUR.shape)
    print(f_mean_LA.shape)
    print(f_mean_OUR.shape)
    print((f_mean_OUR - std_pred_OUR * 2).shape)

    plt.scatter(Xtrain, Ytrain, c="black", zorder=2)
    plt.plot(Xtest, map_pred, linewidth=2, c="green", alpha=1.0, label="MAP", zorder=2)
    plt.plot(Xtest, pred_laplace_mean, linewidth=2, c="yellow", alpha=1.0, label="LA mean", zorder=2)
    plt.plot(Xtest, pred_ours_mean, linewidth=2, c="pink", alpha=1.0, label="Our mean", zorder=2)
    plt.scatter(Xtest_in_region, Ytest_in_region, c="green", zorder=2, marker="x", label="Test points")
    plt.fill_between(
        Xtest.numpy().reshape(-1),
        f_mean_OUR - std_pred_OUR * 2,
        f_mean_OUR + std_pred_OUR * 2,
        alpha=0.3,
        color="tab:blue",
        label="$2\sqrt{\mathbb{V}\,[y]} Our$",
    )
    plt.fill_between(
        Xtest.numpy().reshape(-1),
        f_mean_LA - std_pred_LA * 2,
        f_mean_LA + std_pred_LA * 2,
        alpha=0.3,
        color="tab:orange",
        label="$2\sqrt{\mathbb{V}\,[y]} LA$",
    )
    plt.xlim([-0.5, 6.5])
    plt.ylim([-4, 4])
    plt.legend()
    plt.title("Predictive distribution")
    # plt.savefig("plot_regression_rebuttal/predictive_al.pdf")
    plt.show()

    f_in_region_mean_LA = prediction_in_region_LA.detach().mean(0)
    f_in_region_mean_OUR = prediction_in_region_ours.detach().mean(0)

    # I can compute the variance of the prediction
    f_in_region_var_LA = prediction_in_region_LA.var(0).detach() + la.sigma_noise.square()
    f_in_region_var_OUR = prediction_in_region_ours.var(0).detach() + la.sigma_noise.square()

    # now I think I can compute the nlpd_using_predictions
    _nlpd_LA = nlpd_using_predictions(f_in_region_mean_LA, f_in_region_var_LA, Ytest_in_region)
    _nlpd_OUR = nlpd_using_predictions(f_in_region_mean_OUR, f_in_region_var_OUR, Ytest_in_region)

    # LA_nlpd.append(_nlpd_LA.detach().item())
    # OUR_nlpd.append(_nlpd_OUR.detach().item())
    print("NLPD LA: ", _nlpd_LA)
    print("NLPD ours: ", _nlpd_OUR)

    if args.linearized_pred:
        torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
        with torch.no_grad():
            f_MAP = model2(Xtest)

        with torch.no_grad():
            f_MAP_in_region = model2(Xtest_in_region)

        def predict(params, data):
            y_pred = fmodel(params, buffers, data)
            return y_pred

        pred_ours = torch.zeros((n_posterior_samples, len(Xtest)))
        pred_laplace = torch.zeros((n_posterior_samples, len(Xtest)))
        prediction_in_region_LA = torch.zeros((n_posterior_samples, len(Xtest_in_region)))
        prediction_in_region_ours = torch.zeros((n_posterior_samples, len(Xtest_in_region)))

        for n in range(n_posterior_samples):
            # get the theta weights we are interested in
            w_LA = torch.from_numpy(weights_LA[n, :]).float()
            # put the weights into the model
            torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
            model2.zero_grad()

            diff_weights = w_LA - map_solution

            # now I have to compute the per-sample gradients of this model
            # The jacobian should be computed wrt the map weights
            fmodel, params, buffers = make_functional_with_buffers(model2)

            # I have to make the diff_weights with the same tree shape as the params
            diff_as_params = get_params_structure(diff_weights, params)

            # now I should be able to compute the jvp I need \nabla f_theta (theta - theta_MAP)
            _, jvp_value = jvp(predict, (params, Xtest), (diff_as_params, torch.zeros_like(Xtest)), strict=False)
            # print(jvp_value.shape)

            # now I have the jvp value so I can compute the different functions for
            f_LA = f_MAP + jvp_value

            pred_laplace[n, :] = f_LA.detach().cpu().view(-1)

            # I can try to plot them
            if n == 0:
                plt.plot(Xtest, f_LA.detach().cpu(), linewidth=1, c="red", alpha=0.2, label="LA")
            plt.plot(Xtest, f_LA.detach().cpu(), linewidth=1, c="red", alpha=0.2)

            # maybe I should do the same for the points I want to evaluate
            model2.zero_grad()
            _, jvp_value_in_region = jvp(
                predict, (params, Xtest_in_region), (diff_as_params, torch.zeros_like(Xtest_in_region)), strict=False
            )
            # print(jvp_value.shape)

            # now I have the jvp value so I can compute the different functions for
            f_LA_in_region = f_MAP_in_region + jvp_value_in_region

            prediction_in_region_LA[n, :] = f_LA_in_region.detach().cpu().view(-1)

        # LINEARIZATION IS REALLY AMAZING, CANNOT SAY ANYTHING
        # # at the end I can plot the data
        # plt.scatter(Xtrain, Ytrain, c='black', zorder=2, alpha=0.5)
        # plt.show()

        # I can apply linearization also in our samples
        # IN OUR CASE I THINK WE SHOULD KEEP INTO ACCOUNT THE METRIC WHEN COMPUTING THE LINEARIZATION?
        # OR MAYBE RESCALE SOMETHING
        for n in range(n_posterior_samples):
            # get the theta weights we are interested in
            w_OUR = weights_ours[n, :]  # V_tangent_coord[n, :].reshape(-1, 1)#weights_ours[n,:]
            # put the weights into the model
            torch.nn.utils.vector_to_parameters(map_solution, model2.parameters())
            model2.zero_grad()

            diff_weights = w_OUR - map_solution

            # now I have to compute the per-sample gradients of this model
            # The jacobian should be computed wrt the map weights
            fmodel, params, buffers = make_functional_with_buffers(model2)

            # I have to make the diff_weights with the same tree shape as the params
            diff_as_params = get_params_structure(diff_weights, params)

            # now I should be able to compute the jvp I need \nabla f_theta (theta - theta_MAP)
            _, jvp_value = jvp(predict, (params, Xtest), (diff_as_params, torch.zeros_like(Xtest)), strict=False)
            # print(jvp_value.shape)
            # print(jvp_value.shape)
            # now I have the jvp value so I can compute the different functions for
            f_OUR = f_MAP + jvp_value

            pred_ours[n, :] = f_OUR.detach().cpu().view(-1)

            # print(f_OUR.shape)
            # print('---')
            # I can try to plot them
            if n == 0:
                plt.plot(Xtest, f_OUR.detach().cpu(), linewidth=1, c="blue", alpha=0.2, label="OUR")
            plt.plot(Xtest, f_OUR.detach().cpu(), linewidth=1, c="blue", alpha=0.2)

            # maybe I should do the same for the points I want to evaluate
            model2.zero_grad()
            _, jvp_value_in_region = jvp(
                predict, (params, Xtest_in_region), (diff_as_params, torch.zeros_like(Xtest_in_region)), strict=False
            )

            f_OUR_in_region = f_MAP_in_region + jvp_value_in_region

            prediction_in_region_ours[n, :] = f_OUR_in_region.detach().cpu().view(-1)

        plt.scatter(Xtrain, Ytrain, c="black", zorder=2, alpha=0.5)
        plt.legend()
        plt.xlim([-0.5, 6.5])
        plt.ylim([-4, 4])
        plt.show()

    pred_ours_mean = pred_ours[0:n_posterior_samples, :].mean(0)
    pred_laplace_mean = pred_laplace[0:n_posterior_samples, :].mean(0)

    # now I can consider the predictive
    f_mean_LA = pred_laplace_mean.detach().numpy()
    f_mean_OUR = pred_ours_mean.detach().numpy()

    # I can compute the variance of the prediction
    f_var_LA = pred_laplace.var(0).detach()
    f_var_OUR = pred_ours.var(0).detach()

    std_pred_LA = torch.sqrt(f_var_LA + la.sigma_noise.square()).detach().numpy()
    std_pred_OUR = torch.sqrt(f_var_OUR + la.sigma_noise.square()).detach().numpy()

    plt.scatter(Xtrain, Ytrain, c="black", zorder=2)
    plt.plot(Xtest, map_pred, linewidth=2, c="green", alpha=1.0, label="MAP", zorder=2)
    plt.plot(Xtest, pred_laplace_mean, linewidth=2, c="yellow", alpha=1.0, label="LA mean", zorder=2)
    plt.plot(Xtest, pred_ours_mean, linewidth=2, c="pink", alpha=1.0, label="Our mean", zorder=2)
    plt.scatter(Xtest_in_region, Ytest_in_region, c="green", zorder=2, marker="x", label="Test points")
    plt.fill_between(
        Xtest.numpy().reshape(-1),
        f_mean_OUR - std_pred_OUR * 2,
        f_mean_OUR + std_pred_OUR * 2,
        alpha=0.3,
        color="tab:blue",
        label="$2\sqrt{\mathbb{V}\,[y]} Our$",
    )
    plt.fill_between(
        Xtest.numpy().reshape(-1),
        f_mean_LA - std_pred_LA * 2,
        f_mean_LA + std_pred_LA * 2,
        alpha=0.3,
        color="tab:orange",
        label="$2\sqrt{\mathbb{V}\,[y]} LA$",
    )
    plt.xlim([-0.5, 6.5])
    plt.ylim([-4, 4])
    plt.legend()
    plt.title("Predictive distribution")
    # plt.savefig("plot_regression_rebuttal/predictive_al.pdf")
    plt.show()

    f_in_region_mean_LA = prediction_in_region_LA.detach().mean(0)
    f_in_region_mean_OUR = prediction_in_region_ours.detach().mean(0)

    # I can compute the variance of the prediction
    f_in_region_var_LA = prediction_in_region_LA.var(0).detach() + la.sigma_noise.square()
    f_in_region_var_OUR = prediction_in_region_ours.var(0).detach() + la.sigma_noise.square()

    # now I think I can compute the nlpd_using_predictions
    _nlpd_LA = nlpd_using_predictions(f_in_region_mean_LA, f_in_region_var_LA, Ytest_in_region)
    _nlpd_OUR = nlpd_using_predictions(f_in_region_mean_OUR, f_in_region_var_OUR, Ytest_in_region)

    # LA_nlpd.append(_nlpd_LA.detach().item())
    # OUR_nlpd.append(_nlpd_OUR.detach().item())
    print("NLPD LA linearized: ", _nlpd_LA)
    print("NLPD ours linearized: ", _nlpd_OUR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geomeatric Approximate Inference (GEOMAI)")
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument("--seed", "-s", type=int, default=230, help="seed")
    # parser.add_argument('--dataset', '-data', type=str, default='mnist', help='dataset')
    parser.add_argument("--optimize_prior", "-opt_prior", type=bool, default=False, help="optimize prior")
    parser.add_argument(
        "--optimize_sigma",
        "-opt_sigma",
        type=bool,
        default=False,
        help="optimize sigma (have an effect only if opt is also True)",
    )
    parser.add_argument("--batch_data", "-batch", type=bool, default=False, help="batch data")
    parser.add_argument("--dropout", "-dropout", type=bool, default=False, help="use the model trained with dropout")
    parser.add_argument("--small_model", "-small", type=bool, default=False, help="use the smallest model")
    # parser.add_argument('--load_model', '-load', default=False, type=bool, help='load pretrained model')
    parser.add_argument("--optimizer", "-optim", type=str, default="sgd", help="what optimizer to use")

    parser.add_argument("--structure", "-str", type=str, default="full", help="optimize prior")
    parser.add_argument("--subset", "-sub", type=str, default="all", help="optimize prior")
    parser.add_argument("--samples", "-samp", type=int, default=50, help="number of posterior samples")

    parser.add_argument("--linearized_pred", "-lin", type=bool, default=False, help="Linearization for prediction")
    parser.add_argument(
        "--in_between", "-betw", type=bool, default=False, help="Run model on the in-between uncertainty experiment"
    )

    args = parser.parse_args()
    main(args)
