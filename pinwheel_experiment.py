"""
File with the experiment on the pinwheel dataset.
Generating plot with confidence, where we plot conf = np.max(preds, axis=1).
Compare this with Laplace approximation.
In addition to that, we should measure some classification metrics, like ECE, Brier score, and accuracy.

Update 02/03/2023 @fedbe:

- I am adding the the correction we are doing to our Hessian when we optimize
the prior or when the gradient is not exactly zero.
- I have differentiate between optimizing the prior or not
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("stochman")

from laplace import Laplace
import matplotlib.colors as colors
import seaborn as sns
import geomai.utils.geometry as geometry
from torch import nn
from manifold import cross_entropy_manifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import sklearn.datasets
from datautils import make_pinwheel_data
from utils.metrics import accuracy, nll, brier, calibration
from sklearn.metrics import brier_score_loss
import argparse
from utils.inverse import get_inverse
from torchmetrics.functional.classification import calibration_error
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap
from functorch_utils import get_params_structure, stack_gradient, custum_hvp, stack_gradient2
import os
from datautils import stratified_sampler


def main(args):
    # sns.set_style('darkgrid')
    palette = sns.color_palette("colorblind")

    subset_of_weights = args.subset  #'last_layer' # either 'last_layer' or 'all'
    hessian_structure = args.structure  #'full' # other possibility is 'diag' or 'full'
    n_posterior_samples = args.samples
    security_check = True
    optimize_prior = args.optimize_prior
    print("Are we optimizing the prior? ", optimize_prior)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if device !='cpu':
    #     plot_stuff = False
    #     comput_grid_stuff = False
    batch_data = args.batch_data
    plot_stuff = True
    comput_grid_stuff = True

    # run with several seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_clusters = 5  # number of clusters in pinwheel data
    samples_per_cluster = 200  # number of samples per cluster in pinwheel
    K = 15  # number of components in mixture model
    N = 2  # number of latent dimensions
    P = 2  # number of observation dimensions

    data, labels = make_pinwheel_data(0.3, 0.05, num_clusters, samples_per_cluster, 0.25, seed=42)

    # define train and validation
    X_train = data[:350]
    X_valid = data[350:400]
    X_test = data[400:]

    y_train = labels[:350].astype("int32")
    y_valid = labels[350:400].astype("int32")
    y_test = labels[400:].astype("int32")

    one_hot_yvtest = np.zeros((y_test.size, y_test.max() + 1))
    one_hot_yvtest[np.arange(y_test.size), y_test.reshape(-1)] = 1

    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    X_test = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).long().reshape(-1)
    y_valid = torch.from_numpy(y_valid).long().reshape(-1)
    y_test = torch.from_numpy(y_test).long().reshape(-1)

    X_train = X_train.to(device)
    y_train = y_train.to(device).long()

    X_valid = X_valid.to(device)
    y_valid = y_valid.to(device).long()

    X_test = X_test.to(device)
    y_test = y_test.to(device).long()

    if batch_data:
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

    if plot_stuff:
        plt.scatter(X_train[:, 0], X_train[:, 1], s=40, c=y_train, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        plt.show()

        plt.scatter(X_test[:, 0], X_test[:, 1], s=40, c=y_test, cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]))
        plt.show()

    num_features = X_train.shape[-1]
    print(num_features)
    num_output = num_clusters
    H = 20

    # define the model, I'll start by defining a simple MLP
    # and split it in a feature extractor + last layer only in the
    # case I am doing last-layer approximation

    model = nn.Sequential(
        nn.Linear(num_features, H), torch.nn.Tanh(), nn.Linear(H, H), torch.nn.Tanh(), nn.Linear(H, num_output)
    )
    model.to(device)
    weight_decay = 1e-2
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    # weight_decay = 1e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    r_MAP = 0
    w_b_MAP = 0

    best_valid_accuracy = 0
    max_epoch = 5000  # 1000#2000#2000

    loss_criterion = nn.CrossEntropyLoss(reduction="sum")

    if batch_data:
        for epoch in range(max_epoch):
            for x, y in train_loader:
                y_prob = model(x)
                # print(y_prob.shape)
                # print(y_train.reshape(-1).shape)
                loss = loss_criterion(y_prob, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # now I can evaluate my model on the validation set
            with torch.no_grad():
                valid_probs = model(X_valid)

            valid_pred = torch.argmax(valid_probs, dim=1)
            # print(valid_pred)
            # print(y_valid.reshape(-1))
            # print('----')
            valid_accuracy = (valid_pred == y_valid.view(-1)).int().sum() / len(valid_pred)

            if (epoch + 1) % 100 == 0:
                print(
                    "Epoch: {}, Train loss: {}, Valid acc: {}".format(epoch + 1, loss.detach().item(), valid_accuracy)
                )

            # if valid_accuracy >= best_valid_accuracy:
            #     best_valid_accuracy = valid_accuracy
            #     # store the weights
            #     w_MAP = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            #     # w_b_MAP = torch.nn.utils.parameters_to_vector(W_b.parameters()).detach()

    else:
        for epoch in range(max_epoch):
            y_prob = model(X_train)
            # print(y_prob.shape)
            # print(y_train.reshape(-1).shape)
            loss = loss_criterion(y_prob, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # now I can evaluate my model on the validation set
            # with torch.no_grad():
            #     valid_probs = model(X_valid)

            # valid_pred = torch.argmax(valid_probs, dim=1)
            # # print(valid_pred)
            # # print(y_valid.reshape(-1))
            # # print('----')
            # valid_accuracy = (valid_pred == y_valid.view(-1)).int().sum() / len(valid_pred)

            # if (epoch + 1) % 100 == 0:
            #     print('Epoch: {}, Train loss: {}, Valid acc: {}'.format(epoch+1, loss.detach().item(), valid_accuracy))

            if (epoch + 1) % 100 == 0:
                print("Epoch: {}, Train loss: {}".format(epoch + 1, loss.detach().item()))

            # if valid_accuracy >= best_valid_accuracy:
            #     best_valid_accuracy = valid_accuracy
            # store the weights
            # w_MAP = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            # w_b_MAP = torch.nn.utils.parameters_to_vector(W_b.parameters()).detach()

    w_MAP = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    # at the end of the training I put the best set of weights in the model
    torch.nn.utils.vector_to_parameters(w_MAP, model.parameters())
    print("MAP shape")
    print(w_MAP.shape)
    # create a grid to evaluate our model in terms of confidence
    if comput_grid_stuff:
        N_grid = 100
        offset = 2
        x1min = X_train[:, 0].min() - offset
        x1max = X_train[:, 0].max() + offset
        x2min = X_train[:, 1].min() - offset
        x2max = X_train[:, 1].max() + offset
        # X_grid = utils.my_meshgrid(x1min, x1max, x2min, x2max, N_grid)

        x_grid = torch.linspace(x1min, x1max, N_grid)
        y_grid = torch.linspace(x2min, x2max, N_grid)
        XX1, XX2 = np.meshgrid(x_grid, y_grid)
        X_grid = np.column_stack((XX1.ravel(), XX2.ravel()))

        # computing and plotting the MAP confidence
        with torch.no_grad():
            probs_map = torch.softmax(model(torch.from_numpy(X_grid).float()), dim=1).cpu().numpy()

        conf = probs_map.max(1)

        plt.contourf(
            XX1,
            XX2,
            conf.reshape(N_grid, N_grid),
            alpha=0.8,
            antialiased=True,
            cmap="Blues",
            levels=np.arange(0.1, 1.01, 0.1),
        )
        plt.colorbar()
        plt.scatter(
            X_train[:, 0],
            X_train[:, 1],
            s=45,
            c=y_train,
            edgecolors="black",
            cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]),
        )
        plt.title("Confidence MAP")
        # plt.savefig("pinwheel_classic_plots/confidence_map.pdf")
        plt.show()

    # I have commented because I just need the MAP ECE and MCE because I found it was wrong
    # ok now we can fit the Laplace and our method and check how the confidence changes.
    # Hopefully it should change, as in Kristiadi example
    if not batch_data:
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    # define the Laplace approx
    la = Laplace(
        model,
        "classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        prior_precision=2 * weight_decay,
    )
    la.fit(train_loader)

    if optimize_prior:
        la.optimize_prior_precision(method="marglik")

    print("Prior precision we are using")
    print(la.prior_precision)

    # now I can get some samples for the Laplace approx
    if subset_of_weights == "last_layer":
        if hessian_structure == "diag":
            n_last_layer_weights = num_output * H + num_output
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().numpy()
        else:
            n_last_layer_weights = num_output * H + num_output
            dist = MultivariateNormal(loc=torch.zeros(n_last_layer_weights), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,))
            V_LA = V_LA.detach().numpy()
            print(V_LA.shape)
    else:
        if hessian_structure == "diag":
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().numpy()
            print("V shape")
            print(V_LA.shape)
        else:
            dist = MultivariateNormal(loc=torch.zeros_like(w_MAP), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,))
            V_LA = V_LA.detach().cpu().numpy()
            print(V_LA.shape)

    if subset_of_weights == "last_layer":
        weights_ours = torch.zeros(n_posterior_samples, len(w_MAP))
        # weights_LA = torch.zeros(n_posterior_samples, len(w_MAP))

        MAP = w_MAP.clone()
        feature_extractor_map = MAP[0:-n_last_layer_weights]
        ll_map = MAP[-n_last_layer_weights:]
        print(feature_extractor_map.shape)
        print(ll_map.shape)

        # and now I have to define again the model
        feature_extractor_model = torch.nn.Sequential(
            nn.Linear(num_features, H), torch.nn.Tanh(), nn.Linear(H, H), torch.nn.Tanh()
        )
        ll = nn.Linear(H, num_output)

        # and use the correct weights
        torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
        torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

        # I have to precompute some stuff
        # i.e. I am treating the hidden activation before the last layer as my input
        # because since the weights are fixed, then this feature vector is fixed
        with torch.no_grad():
            R = feature_extractor_model(X_train)

        if optimize_prior:
            manifold = cross_entropy_manifold(ll, R, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2)

        else:
            manifold = cross_entropy_manifold(ll, R, y_train, batching=False, lambda_reg=weight_decay)

    else:
        model2 = nn.Sequential(
            nn.Linear(num_features, H), torch.nn.Tanh(), nn.Linear(H, H), torch.nn.Tanh(), nn.Linear(H, num_output)
        )
        model2.to(device)
        # here depending if I am using a diagonal approx, I have to redefine the model
        if optimize_prior:
            if batch_data:
                manifold = cross_entropy_manifold(
                    model2,
                    train_loader,
                    y=None,
                    batching=True,
                    lambda_reg=la.prior_precision.item() / 2,
                    device=device,
                )

            else:
                manifold = cross_entropy_manifold(
                    model2, X_train, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2, device=device
                )
        else:
            if batch_data:
                manifold = cross_entropy_manifold(
                    model2, train_loader, y=None, batching=True, lambda_reg=weight_decay, device=device
                )

            else:
                manifold = cross_entropy_manifold(
                    model2, X_train, y_train, batching=False, lambda_reg=weight_decay, device=device
                )

    with torch.no_grad():
        if subset_of_weights == "last_layer":
            print("what?")
            print(ll_map.shape)
            grad = manifold.get_gradient_value_in_specific_point(ll_map.clone())
        else:
            print("Shape of w_MAP")
            print(w_MAP.shape)
            grad = manifold.get_gradient_value_in_specific_point(w_MAP.clone()).detach()

    # now I have everything to compute the expmap and store the weights
    weights_ours = torch.zeros(n_posterior_samples, len(w_MAP)).to(device)

    for n in tqdm(range(n_posterior_samples), desc="Solving expmap"):
        v = V_LA[n, :].reshape(-1, 1)  # V_LA[n, :].reshape(-1, 1)

        # here I have to add the using batches to solve the computation

        if subset_of_weights == "last_layer":
            curve, failed = geometry.expmap(manifold, ll_map.clone(), v)
            if failed:
                pass
            _new_ll_weights = curve(1)[0]
            _new_weights = torch.cat(
                (feature_extractor_map.view(-1), torch.from_numpy(_new_ll_weights).float().view(-1)), dim=0
            )
            weights_ours[n, :] = _new_weights.view(-1)
            torch.nn.utils.vector_to_parameters(_new_weights, model.parameters())

        else:
            if args.expmap_different_batches:
                n_sub_data = args.subset_data_size
                # idx_sub = np.random.choice(np.arange(0,len(X_train),1),n_sub_data)
                idx_sub = stratified_sampler(y_train, n_sub_data, num_clusters)
                sub_x_train = X_train[idx_sub, :]
                sub_y_train = y_train[idx_sub]

                # I can define the new manifold
                manifold = cross_entropy_manifold(
                    model2,
                    sub_x_train,
                    sub_y_train,
                    batching=False,
                    lambda_reg=la.prior_precision.item() / 2,
                    N=len(X_train),
                    B1=n_sub_data,
                    device=device,
                )

            curve, failed = geometry.expmap(manifold, w_MAP.clone(), v)
            if failed:
                print("FAIL")
                pass
            _new_weights = curve(1)[0]
            weights_ours[n, :] = torch.from_numpy(_new_weights.reshape(-1)).to(device)

    ############ LAPLACE

    # I can also directly create the weights, by adding the map
    weights_LA = torch.zeros(n_posterior_samples, len(w_MAP)).to(device)
    # now I have to define the manifolds. While it's straihgtforward for the full network case
    # we have to modify the network for the last layer case

    for n in range(n_posterior_samples):
        if subset_of_weights == "last_layer":
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float() + ll_map.clone()
            laplace_weigths = torch.cat((feature_extractor_map.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths.cpu()
        else:
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float().to(device) + w_MAP
            # laplace_weigths = torch.cat((feature_extractor_MAP.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths.cpu()

    if comput_grid_stuff:
        P_grid_LAPLACE = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
            # compute the predictions
            with torch.no_grad():
                P_grid_LAPLACE += torch.softmax(model(torch.from_numpy(X_grid).float()), dim=1).numpy()

        P_grid_LAPLACE /= n_posterior_samples
        P_grid_LAPLACE_conf = P_grid_LAPLACE.max(1)
        print(P_grid_LAPLACE_conf)
        print(np.min(P_grid_LAPLACE_conf))
        print(np.max(P_grid_LAPLACE_conf))
        plt.contourf(
            XX1,
            XX2,
            P_grid_LAPLACE_conf.reshape(N_grid, N_grid),
            alpha=0.7,
            antialiased=True,
            cmap="Blues",
            levels=np.arange(0.0, 1.01, 0.1),
        )
        # plt.colorbar()
        plt.scatter(
            X_train[:, 0],
            X_train[:, 1],
            s=45,
            c=y_train,
            edgecolors="k",
            cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]),
        )
        # plt.title('All weights, full Hessian approx - Confidence LA')
        # plt.savefig("pinwheel_classic_plots/confidence_la_al_prior_not_optim.pdf")
        plt.show()

        # and then our stuff
        P_grid_OUR = 0
        for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
            # put the weights in the model
            torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
            # compute the predictions
            with torch.no_grad():
                P_grid_OUR += torch.softmax(model(torch.from_numpy(X_grid).float()), dim=1).numpy()

        P_grid_OUR /= n_posterior_samples
        P_grid_OUR_conf = P_grid_OUR.max(1)
        print(P_grid_OUR_conf)
        print(np.min(P_grid_OUR_conf))
        print(np.max(P_grid_OUR_conf))
        plt.contourf(
            XX1,
            XX2,
            P_grid_OUR_conf.reshape(N_grid, N_grid),
            alpha=0.8,
            antialiased=True,
            cmap="Blues",
            levels=np.arange(0.0, 1.01, 0.1),
        )
        # plt.colorbar()
        plt.scatter(
            X_train[:, 0],
            X_train[:, 1],
            s=45,
            c=y_train,
            edgecolors="black",
            cmap=colors.ListedColormap(plt.cm.tab10.colors[:5]),
        )
        # plt.title('All weights, full Hessian approx - Confidence OURS')
        # plt.savefig("pinwheel_classic_plots/confidence_our_al_prior_not_optim.pdf")
        plt.show()

    # I have to compute the prediction for the xtest
    py_test_LA = 0
    for n in tqdm(range(n_posterior_samples), desc="computing laplace prediction in region"):
        # put the weights in the model
        torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
        # compute the predictions
        with torch.no_grad():
            py_test_LA += torch.softmax(model(X_test), dim=1)

    py_test_OUR = 0
    for n in tqdm(range(n_posterior_samples), desc="computing laplace prediction in region"):
        # put the weights in the model
        torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
        # compute the predictions
        with torch.no_grad():
            py_test_OUR += torch.softmax(model(X_test), dim=1)

    py_test_OUR /= n_posterior_samples
    py_test_LA /= n_posterior_samples

    accuracy_LA = accuracy(py_test_LA, y_test)
    accuracy_OURS = accuracy(py_test_OUR, y_test)

    nll_LA = nll(py_test_LA, y_test)
    nll_OUR = nll(py_test_OUR, y_test)

    brier_LA = brier(py_test_LA, y_test)
    brier_OURS = brier(py_test_OUR, y_test)

    ece_la = calibration_error(py_test_LA, y_test, norm="l1", task="multiclass", num_classes=5, n_bins=15) * 100
    mce_la = calibration_error(py_test_LA, y_test, norm="max", task="multiclass", num_classes=5, n_bins=15) * 100

    ece_our = calibration_error(py_test_OUR, y_test, norm="l1", task="multiclass", num_classes=5, n_bins=15) * 100
    mce_our = calibration_error(py_test_OUR, y_test, norm="max", task="multiclass", num_classes=5, n_bins=15) * 100

    # I can also consider the MAP and check its calibration
    torch.nn.utils.vector_to_parameters(w_MAP, model.parameters())
    with torch.no_grad():
        py_test_MAP = torch.softmax(model(X_test), dim=1)

    accuracy_MAP = accuracy(py_test_MAP, y_test)

    nll_MAP = nll(py_test_MAP, y_test)

    brier_MAP = brier(py_test_MAP, y_test)

    ece_map = calibration_error(py_test_MAP, y_test, norm="l1", task="multiclass", num_classes=5, n_bins=15) * 100
    mce_map = calibration_error(py_test_MAP, y_test, norm="max", task="multiclass", num_classes=5, n_bins=15) * 100

    print(f"Results MAP: accuracy {accuracy_MAP}, nll {nll_MAP}, brier {brier_MAP}, ECE {ece_map}, MCE {mce_map}")
    print(f"Results LA: accuracy {accuracy_LA}, nll {nll_LA}, brier {brier_LA}, ECE {ece_la}, MCE {mce_la}")
    print(f"Results OURS: accuracy {accuracy_OURS}, nll {nll_OUR}, brier {brier_OURS}, ECE {ece_our}, MCE {mce_our}")

    # RESULTS_FOLDER = "pinwheel_exp_final/"
    # if not os.path.exists(RESULTS_FOLDER):
    #     os.makedirs(RESULTS_FOLDER)

    if args.expmap_different_batches:
        results_file_name = f"results_banana_{args.seed}_optim_sgd_prior_optim_{args.optimize_prior}_linearization_{args.linearized_pred}_expmap_computed_with_{n_sub_data}_data.pt"
    else:
        results_file_name = f"results_banana_{args.seed}_optim_sgd_prior_optim_{args.optimize_prior}_linearization_{args.linearized_pred}.pt"

        # now I can create my dictionary
    dict_MAP = {"Accuracy": accuracy_MAP, "NLL": nll_MAP, "Brier": brier_MAP, "ECE": ece_map, "MCE": mce_map}
    dict_LA = {"Accuracy": accuracy_LA, "NLL": nll_LA, "Brier": brier_LA, "ECE": ece_la, "MCE": mce_la}
    dict_OUR = {"Accuracy": accuracy_OURS, "NLL": nll_OUR, "Brier": brier_OURS, "ECE": ece_our, "MCE": mce_our}

    final_dict = {"results_MAP": dict_MAP, "results_LA": dict_LA, "results_OUR": dict_OUR}

    # torch.save(final_dict, RESULTS_FOLDER + results_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geomeatric Approximate Inference (GEOMAI)")
    # parser.add_argument('--seed', '-seed', type=int, default=0, help='seed for randomness generator')
    parser.add_argument("--seed", "-s", type=int, default=230, help="seed")
    # parser.add_argument('--dataset', '-data', type=str, default='mnist', help='dataset')
    parser.add_argument("--optimize_prior", "-opt_prior", type=bool, default=False, help="optimize prior")
    # parser.add_argument('--optimize_sigma', '-opt_sigma', type=bool, default=False, help='optimize sigma (have an effect only if opt is also True)')
    parser.add_argument("--batch_data", "-batch", type=bool, default=False, help="batch data")
    # parser.add_argument('--load_model', '-load', default=False, type=bool, help='load pretrained model')
    parser.add_argument("--structure", "-str", type=str, default="full", help="Hessian struct for Laplace")
    parser.add_argument("--subset", "-sub", type=str, default="all", help="subset of weights for Laplace")
    parser.add_argument("--samples", "-samp", type=int, default=50, help="number of posterior samples")
    parser.add_argument("--linearized_pred", "-lin", type=bool, default=False, help="Linearization for prediction")
    parser.add_argument(
        "--expmap_different_batches",
        "-batches",
        type=bool,
        default=False,
        help="Solve exponential map using only a batch of the data and not the full dataset",
    )
    parser.add_argument(
        "--subset_data_size",
        "-subset_data",
        type=int,
        default=150,
        help="Number of datapoints in the batch we should consider",
    )

    args = parser.parse_args()
    main(args)
