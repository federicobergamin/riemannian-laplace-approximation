"""
File containing different functions to load different data
"""
import torch
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, EMNIST, KMNIST
from torchvision import transforms
import os
import sklearn
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler


class RotationTransform:
    """Rotate the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate, seed):
    np.random.seed(seed)
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    rotated_feat = np.einsum("ti,tij->tj", features, rotations)
    data_plus_labels = np.concatenate((rotated_feat, labels.reshape(-1, 1)), axis=1)

    permuted_dataset = np.random.permutation(data_plus_labels)

    data = permuted_dataset[:, 0:2]
    shuffled_labels = permuted_dataset[:, 2:]
    # return 10*np.random.permutation(np.einsum('ti,tij->tj', features, rotations))
    return data, shuffled_labels


def get_MNIST(
    DATA_DIR="/scratch/",
    batch_size=128,
    for_ood_eval=False,
    model_type="CNN",
    apply_PCA=False,
    only_three_classes=False,
):
    if apply_PCA:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )
    elif model_type == "CNN":
        # using CNN here
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

    train_dataset = MNIST(root=DATA_DIR, train=True, transform=transform, download=True)

    valid_test_dataset = MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    if for_ood_eval:
        # I have to return the test set and then split it into
        # validation and test set
        test_loader = torch.utils.data.DataLoader(valid_test_dataset, batch_size=batch_size, shuffle=False)

        return test_loader

    else:
        if only_three_classes:
            idx_classes_train = (
                (train_dataset.targets == 0) | (train_dataset.targets == 1) | (train_dataset.targets == 2)
            )
            train_dataset.targets = train_dataset.targets[idx_classes_train]
            train_dataset.data = train_dataset.data[idx_classes_train]

            idx_classes_valid_test = (
                (valid_test_dataset.targets == 0)
                | (valid_test_dataset.targets == 1)
                | (valid_test_dataset.targets == 2)
            )
            valid_test_dataset.targets = valid_test_dataset.targets[idx_classes_valid_test]
            valid_test_dataset.data = valid_test_dataset.data[idx_classes_valid_test]

        N_valid_examples = int(0.1 * valid_test_dataset.data.shape[0])  # if only_three_classes else 2000
        N_test_examples = valid_test_dataset.data.shape[0] - N_valid_examples

        valid_dataset, test_dataset = torch.utils.data.random_split(
            valid_test_dataset, [N_valid_examples, N_test_examples], generator=torch.Generator().manual_seed(42)
        )

        if apply_PCA:
            batch_size_pca = 1
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_pca, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_pca, shuffle=False)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size_pca, shuffle=False)

            # now I have to get my images and and labels for both the daataset
            # and apply the PCA
            NOF = len(train_dataset)
            X_train = torch.zeros((NOF, 784))
            y_train = torch.zeros(NOF)
            for i, (batch_x, batch_y) in enumerate(train_loader):
                idx_start = i * batch_size
                idx_end = idx_start + len(batch_x)
                X_train[idx_start:idx_end, :] = batch_x
                y_train[idx_start:idx_end] = batch_y

            # I should do the same for validation and test
            X_valid = torch.zeros((N_valid_examples, 784))
            y_valid = torch.zeros(N_valid_examples)
            for i, (batch_x, batch_y) in enumerate(valid_loader):
                idx_start = i * batch_size
                idx_end = idx_start + len(batch_x)
                X_valid[idx_start:idx_end, :] = batch_x
                y_valid[idx_start:idx_end] = batch_y

            X_test = torch.zeros((N_test_examples, 784))
            y_test = torch.zeros(N_test_examples)
            for i, (batch_x, batch_y) in enumerate(test_loader):
                idx_start = i * batch_size
                idx_end = idx_start + len(batch_x)
                X_test[idx_start:idx_end, :] = batch_x
                y_test[idx_start:idx_end] = batch_y

            print("X_train shape is ", X_train.shape)
            print("X_valid shape is ", X_valid.shape)
            print("X_test shape is ", X_test.shape)
            print("y_train shape is ", y_train.shape)
            print("y_valid shape is ", y_valid.shape)
            print("y_test shape is ", y_test.shape)
            print(torch.unique(y_train))
            print(torch.unique(y_valid))
            print(torch.unique(y_test))

            # now I have to compute PCA on these
            from sklearn.decomposition import PCA

            n_components = 5
            pca = PCA(n_components=n_components)
            pca.fit(X_train)

            # transform the data
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            X_valid = pca.transform(X_valid)

            print("PCA X_train shape is ", X_train.shape)
            print("PCA X_valid shape is ", X_valid.shape)
            print("PCA X_test shape is ", X_test.shape)

            train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), y_train.long())
            test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), y_test.long())
            valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_valid).float(), y_valid.long())

            # and dataloaders
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader


def get_FMNIST(DATA_DIR="/scratch/", batch_size=128, for_ood_eval=False, model_type="CNN"):
    if model_type == "CNN":
        # using CNN here
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

    train_dataset = FashionMNIST(root=DATA_DIR, train=True, transform=transform, download=True)

    valid_test_dataset = FashionMNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    if for_ood_eval:
        # create dataloader
        test_loader = torch.utils.data.DataLoader(valid_test_dataset, batch_size=batch_size, shuffle=False)
        return test_loader
    else:
        valid_dataset, test_dataset = torch.utils.data.random_split(
            valid_test_dataset, [2000, 8000], generator=torch.Generator().manual_seed(42)
        )

        batch_size = 128
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader


def get_EMNIST(DATA_DIR="/scratch/", for_ood_eval=False, batch_size=128, model_type="CNN"):
    if model_type == "CNN":
        # using CNN here
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

    if for_ood_eval:
        emnist_val_test_set = EMNIST(DATA_DIR, split="digits", train=False, transform=transform, download=True)

        # create dataloader
        test_loader = torch.utils.data.DataLoader(emnist_val_test_set, batch_size=batch_size, shuffle=False)

    else:
        raise NotImplementedError("EMNIST is not implemented for in-distribution evaluation")

    return test_loader


def get_KMNIST(DATA_DIR="/scratch/", for_ood_eval=False, batch_size=128, model_type="CNN"):
    if model_type == "CNN":
        # using CNN here
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

    if for_ood_eval:
        kmnist_val_test_set = KMNIST(DATA_DIR, train=False, transform=transform, download=True)

        # create dataloader
        test_loader = torch.utils.data.DataLoader(kmnist_val_test_set, batch_size=batch_size, shuffle=False)

    else:
        raise NotImplementedError("KMNIST is not implemented for in-distribution evaluation")
    return test_loader


def get_rotated_MNIST(DATA_DIR="/scratch/", for_ood_eval=False, batch_size=128, model_type="CNN", angle=90):
    # rotation angles that people are considering [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    if model_type == "CNN":
        # using CNN here
        transform = transforms.Compose([RotationTransform(angle), transforms.ToTensor()])
    else:
        transform = transforms.Compose(
            [RotationTransform(angle), transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
        )

    if for_ood_eval:
        rotated_mnist_val_test_set = MNIST(DATA_DIR, train=False, transform=transform, download=True)

        # create dataloader
        test_loader = torch.utils.data.DataLoader(rotated_mnist_val_test_set, batch_size=batch_size, shuffle=False)

    else:
        raise NotImplementedError("rotated-MNIST is not implemented for in-distribution evaluation")
    return test_loader


def get_ood_loader(dataset="mnist", batch_size=128, model_type="CNN", angle=90):
    if dataset == "mnist":
        return get_MNIST(DATA_DIR="/scratch/", for_ood_eval=True, batch_size=batch_size, model_type=model_type)
    elif dataset == "fmnist":
        return get_FMNIST(DATA_DIR="/scratch/", for_ood_eval=True, batch_size=batch_size, model_type=model_type)
    elif dataset == "emnist":
        return get_EMNIST(DATA_DIR="/scratch/", for_ood_eval=True, batch_size=batch_size, model_type=model_type)
    elif dataset == "kmnist":
        return get_KMNIST(DATA_DIR="/scratch/", for_ood_eval=True, batch_size=batch_size, model_type=model_type)
    elif dataset == "rotated_mnist":
        return get_rotated_MNIST(
            DATA_DIR="/scratch/", for_ood_eval=True, batch_size=batch_size, model_type=model_type, angle=angle
        )
    else:
        raise NotImplementedError("Dataset not implemented")


def get_sinusoid_example(n_data=150, sigma_noise=0.3, batch_size=150):
    # create simple sinusoid data set
    X_train = (torch.rand(n_data) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma_noise
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size)
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test


# I have to create a method to get subsampled MNIST and subsamplet FMNIST
def get_subsampled_MNIST(
    DATA_DIR="/scratch/", n_examples_per_labels=500, batch_size=128, return_data_array=False, type_network="cnn"
):
    if type_network == "fc":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(root=DATA_DIR, train=True, transform=transform, download=True)

    valid_test_dataset = MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    # now I have to operate with the training set
    # I have to select only 500 examples for each class
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # note h
    if type_network == "fc":
        NOF = len(train_dataset)
        X_train = torch.zeros((NOF, 784))
        y_train = torch.zeros(NOF)
        for i, (batch_x, batch_y) in enumerate(train_loader):
            idx_start = i * batch_size
            idx_end = idx_start + len(batch_x)
            X_train[idx_start:idx_end, :] = batch_x
            y_train[idx_start:idx_end] = batch_y

        # now I have my transformed dataset
        new_X_train = torch.zeros((n_examples_per_labels * 10, 784))
        new_y_train = torch.zeros(n_examples_per_labels * 10).long()

        sums = 0
        for i in range(10):
            # get all the indices of a specific label
            ids_i = torch.where(y_train == i)[0]
            # print(len(ids_i))
            # sums +=len(ids_i)
            data_i = X_train[ids_i, :]
            label_i = y_train[ids_i]
            # print(label_i)
            # print('-----')
            sub_data = data_i[0:n_examples_per_labels, :]
            sub_y = label_i[0:n_examples_per_labels]

            # print('sub data: ', sub_data.shape)
            # print('sub y:', sub_y.shape)
            # I have to add this to the new dataset
            idx_start = i * n_examples_per_labels
            idx_end = idx_start + len(sub_data)
            # print(f'idx start {idx_start}')
            # print(f'idx end {idx_end}')
            new_X_train[idx_start:idx_end, :] = sub_data
            new_y_train[idx_start:idx_end] = sub_y.long()

    else:
        # here we have to deal with CNN style
        NOF = len(train_dataset)
        y_train = torch.zeros(NOF)
        X_train = torch.zeros((NOF, 1, 28, 28))
        for i, (batch_x, batch_y) in enumerate(train_loader):
            idx_start = i * batch_size
            idx_end = idx_start + len(batch_x)
            X_train[idx_start:idx_end, :, :] = batch_x
            y_train[idx_start:idx_end] = batch_y

        # now I have my transformed dataset
        new_X_train = torch.zeros((n_examples_per_labels * 10, 1, 28, 28))
        new_y_train = torch.zeros(n_examples_per_labels * 10).long()

        for i in range(10):
            # get all the indices of a specific label
            ids_i = torch.where(y_train == i)[0]
            # print(len(ids_i))
            # sums +=len(ids_i)
            data_i = X_train[ids_i, :, :]
            label_i = y_train[ids_i]
            # print(label_i)
            # print('-----')
            sub_data = data_i[0:n_examples_per_labels, :, :]
            sub_y = label_i[0:n_examples_per_labels]

            # print('sub data: ', sub_data.shape)
            # print('sub y:', sub_y.shape)
            # I have to add this to the new dataset
            idx_start = i * n_examples_per_labels
            idx_end = idx_start + len(sub_data)
            # print(f'idx start {idx_start}')
            # print(f'idx end {idx_end}')
            new_X_train[idx_start:idx_end, :, :] = sub_data
            new_y_train[idx_start:idx_end] = sub_y.long()

    # print(f'total sum is {sums}')
    # now I can create a new dataset and dataloader
    new_train_dataset = torch.utils.data.TensorDataset(new_X_train, new_y_train)
    new_train_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True)

    # now I can create test and validation set
    N_valid_examples = 2000  # if only_three_classes else 2000
    N_test_examples = valid_test_dataset.data.shape[0] - N_valid_examples
    valid_dataset, test_dataset = torch.utils.data.random_split(
        valid_test_dataset, [N_valid_examples, N_test_examples], generator=torch.Generator().manual_seed(42)
    )

    # and crate the two splits
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if return_data_array:
        return new_train_loader, valid_loader, test_loader, new_X_train, new_y_train
    else:
        return new_train_loader, valid_loader, test_loader


# I can create the same for FMNIST
def get_subsampled_FMNIST(
    DATA_DIR="/scratch/", n_examples_per_labels=500, batch_size=128, return_data_array=False, type_network="cnn"
):
    if type_network == "fc":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = FashionMNIST(root=DATA_DIR, train=True, transform=transform, download=True)

    valid_test_dataset = FashionMNIST(root=DATA_DIR, train=False, transform=transform, download=True)

    # now I have to operate with the training set
    # I have to select only 500 examples for each class
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # note h
    if type_network == "fc":
        NOF = len(train_dataset)
        X_train = torch.zeros((NOF, 784))
        y_train = torch.zeros(NOF)
        for i, (batch_x, batch_y) in enumerate(train_loader):
            idx_start = i * batch_size
            idx_end = idx_start + len(batch_x)
            X_train[idx_start:idx_end, :] = batch_x
            y_train[idx_start:idx_end] = batch_y

        # now I have my transformed dataset
        new_X_train = torch.zeros((n_examples_per_labels * 10, 784))
        new_y_train = torch.zeros(n_examples_per_labels * 10).long()

        sums = 0
        for i in range(10):
            # get all the indices of a specific label
            ids_i = torch.where(y_train == i)[0]
            # print(len(ids_i))
            # sums +=len(ids_i)
            data_i = X_train[ids_i, :]
            label_i = y_train[ids_i]
            # print(label_i)
            # print('-----')
            sub_data = data_i[0:n_examples_per_labels, :]
            sub_y = label_i[0:n_examples_per_labels]

            # print('sub data: ', sub_data.shape)
            # print('sub y:', sub_y.shape)
            # I have to add this to the new dataset
            idx_start = i * n_examples_per_labels
            idx_end = idx_start + len(sub_data)
            # print(f'idx start {idx_start}')
            # print(f'idx end {idx_end}')
            new_X_train[idx_start:idx_end, :] = sub_data
            new_y_train[idx_start:idx_end] = sub_y.long()

    else:
        # here we have to deal with CNN style
        NOF = len(train_dataset)
        y_train = torch.zeros(NOF)
        X_train = torch.zeros((NOF, 1, 28, 28))
        for i, (batch_x, batch_y) in enumerate(train_loader):
            idx_start = i * batch_size
            idx_end = idx_start + len(batch_x)
            X_train[idx_start:idx_end, :, :] = batch_x
            y_train[idx_start:idx_end] = batch_y

        # now I have my transformed dataset
        new_X_train = torch.zeros((n_examples_per_labels * 10, 1, 28, 28))
        new_y_train = torch.zeros(n_examples_per_labels * 10).long()

        for i in range(10):
            # get all the indices of a specific label
            ids_i = torch.where(y_train == i)[0]
            # print(len(ids_i))
            # sums +=len(ids_i)
            data_i = X_train[ids_i, :, :]
            label_i = y_train[ids_i]
            # print(label_i)
            # print('-----')
            sub_data = data_i[0:n_examples_per_labels, :, :]
            sub_y = label_i[0:n_examples_per_labels]

            # print('sub data: ', sub_data.shape)
            # print('sub y:', sub_y.shape)
            # I have to add this to the new dataset
            idx_start = i * n_examples_per_labels
            idx_end = idx_start + len(sub_data)
            # print(f'idx start {idx_start}')
            # print(f'idx end {idx_end}')
            new_X_train[idx_start:idx_end, :, :] = sub_data
            new_y_train[idx_start:idx_end] = sub_y.long()

    # now I can create a new dataset and dataloader
    new_train_dataset = torch.utils.data.TensorDataset(new_X_train, new_y_train)
    new_train_loader = torch.utils.data.DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True)

    # now I can create test and validation set
    N_valid_examples = 2000  # if only_three_classes else 2000
    N_test_examples = valid_test_dataset.data.shape[0] - N_valid_examples
    valid_dataset, test_dataset = torch.utils.data.random_split(
        valid_test_dataset, [N_valid_examples, N_test_examples], generator=torch.Generator().manual_seed(42)
    )

    # and crate the two splits
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if return_data_array:
        return new_train_loader, valid_loader, test_loader, new_X_train, new_y_train
    else:
        return new_train_loader, valid_loader, test_loader


def get_subsampled_dataset(
    DATA_DIR="/scratch/",
    dataset="mnist",
    n_examples_per_labels=500,
    batch_size=128,
    return_data_array=False,
    type_network="cnn",
):
    assert dataset in ["mnist", "fmnist"]

    if return_data_array:
        if dataset == "mnist":
            train_loader, valid_loader, test_loader, new_X, new_y = get_subsampled_MNIST(
                DATA_DIR=DATA_DIR,
                n_examples_per_labels=n_examples_per_labels,
                batch_size=batch_size,
                return_data_array=return_data_array,
                type_network=type_network,
            )
        else:
            train_loader, valid_loader, test_loader, new_X, new_y = get_subsampled_FMNIST(
                DATA_DIR=DATA_DIR,
                n_examples_per_labels=n_examples_per_labels,
                batch_size=batch_size,
                return_data_array=return_data_array,
                type_network=type_network,
            )

        return train_loader, valid_loader, test_loader, new_X, new_y
    else:
        if dataset == "mnist":
            train_loader, valid_loader, test_loader, new_X, new_y = get_subsampled_MNIST(
                DATA_DIR=DATA_DIR,
                n_examples_per_labels=n_examples_per_labels,
                batch_size=batch_size,
                return_data_array=return_data_array,
                type_network=type_network,
            )
        else:
            train_loader, valid_loader, test_loader, new_X, new_y = get_subsampled_FMNIST(
                DATA_DIR=DATA_DIR,
                n_examples_per_labels=n_examples_per_labels,
                batch_size=batch_size,
                return_data_array=return_data_array,
                type_network=type_network,
            )

        return train_loader, valid_loader, test_loader


def stratified_sampler(ytest, n_samples, n_classes=10):
    n_sample_per_classes = int(n_samples / n_classes)

    # now I have to randomly pick the indices of each label
    indeces_to_return = []
    for i in range(n_classes):
        idx_class = torch.where(ytest == i)[0]
        idx_class = idx_class.cpu().numpy()
        idx_for_the_class = np.random.choice(idx_class, n_sample_per_classes)
        indeces_to_return.extend(idx_for_the_class)
    indeces_to_return = np.array(indeces_to_return)

    return indeces_to_return


class UCIClassificationDatasets(data.Dataset):
    def __init__(
        self,
        data_set,
        split_train_size=0.7,
        random_seed=6,
        shuffle=True,
        stratify=None,
        root="data/UCI/",
        train=True,
        valid=False,
        scaling=True,
        double=False,
    ):
        assert isinstance(random_seed, int), "Please provide an integer random seed"
        error_msg = "invalid UCI classification dataset"
        assert data_set in [
            "australian",
            "breast_cancer",
            "glass",
            "ionosphere",
            "vehicle",
            "waveform",
            "satellite",
            "digits",
            "banana",
        ], error_msg

        assert isinstance(split_train_size, float), "split_train_size can only be float"

        assert 0.0 <= split_train_size <= 1.0, "split_train_size does not lie between 0 and 1"

        self.root = root
        self.train = train
        self.valid = valid
        if data_set in ["australian", "breast_cancer"]:
            if data_set == "australian":
                aus = "australian_presplit"
                x_train_file = os.path.join(self.root, aus, "australian_scale_X_tr.csv")
                x_test_file = os.path.join(self.root, aus, "australian_scale_X_te.csv")
                y_train_file = os.path.join(self.root, aus, "australian_scale_y_tr.csv")
                y_test_file = os.path.join(self.root, aus, "australian_scale_y_te.csv")
            else:
                bca = "breast_cancer_scale_presplit"
                x_train_file = os.path.join(self.root, bca, "breast_cancer_scale_X_tr.csv")
                x_test_file = os.path.join(self.root, bca, "breast_cancer_scale_X_te.csv")
                y_train_file = os.path.join(self.root, bca, "breast_cancer_scale_y_tr.csv")
                y_test_file = os.path.join(self.root, bca, "breast_cancer_scale_y_te.csv")

            x_train, x_test = np.loadtxt(x_train_file), np.loadtxt(x_test_file)
            y_train, y_test = np.loadtxt(y_train_file), np.loadtxt(y_test_file)

        elif data_set == "ionosphere":
            # hacky setting x_train, x_test
            filen = os.path.join(self.root, data_set, "ionosphere.data")
            Xy = np.loadtxt(filen, delimiter=",")
            x_train, x_test = Xy[:50, :-1], Xy[50:, :-1]
            y_train, y_test = Xy[:50, -1], Xy[50:, -1]

        elif data_set == "banana":
            # hacky setting x_train, x_test
            filen = os.path.join(self.root, data_set, "banana.csv")
            Xy = np.loadtxt(filen, delimiter=",")
            x_train, y_train = Xy[:, :-1], Xy[:, -1]
            x_test, y_test = Xy[:0, :-1], Xy[:0, -1]
            y_train, y_test = y_train - 1, y_test - 1

        elif data_set == "digits":
            # hacky setting x_train, x_test
            from sklearn.datasets import load_digits

            X, y = load_digits(return_X_y=True)
            x_train, x_test = X[:50], X[50:]
            y_train, y_test = y[:50], y[50:]

        else:
            # hacky setting x_train, x_test
            x_file = os.path.join(self.root, data_set, "X.txt")
            y_file = os.path.join(self.root, data_set, "Y.txt")
            X = np.loadtxt(x_file)
            y = np.loadtxt(y_file)
            x_train, x_test = X[:50], X[50:]
            y_train, y_test = y[:50], y[50:]

        x_full, y_full = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
        strat = y_full if stratify else None
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x_full, y_full, train_size=split_train_size, random_state=random_seed, shuffle=shuffle, stratify=strat
        )
        strat = y_test if stratify else None
        x_test, x_valid, y_test, y_valid = sklearn.model_selection.train_test_split(
            x_test, y_test, train_size=0.5, random_state=random_seed, shuffle=shuffle, stratify=strat
        )
        assert (len(y_test) + len(y_valid) + len(y_train)) == len(y_full)
        assert (len(x_test) + len(x_valid) + len(x_train)) == len(x_full)

        if scaling:
            self.scl = StandardScaler(copy=False)
            self.scl.fit_transform(x_train)
            self.scl.transform(x_test)
            self.scl.transform(x_valid)

        # impossible setting: if train is false, valid needs to be false too
        assert not (self.train and self.valid)
        if self.train:
            self.data, self.targets = torch.from_numpy(x_train), torch.from_numpy(y_train)
        else:
            if self.valid:
                self.data, self.targets = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
            else:
                self.data, self.targets = torch.from_numpy(x_test), torch.from_numpy(y_test)

        if double:
            self.data = self.data.double()
            self.targets = self.targets.double()
        else:
            self.data = self.data.float()
            self.targets = self.targets.float()

        self.C = len(self.targets.unique())
        # for multiclass
        if self.C > 2:
            self.targets = self.targets.long()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def sampling_with_same_proportions(ytest, n_samples, n_classes=2):
    if n_classes > 2:
        raise NotImplementedError("This naive implementation for now only work for 2 classes")
    n_class1 = len(np.where(ytest == 0)[0])
    n_class2 = len(np.where(ytest == 1)[0])

    proportion_class1 = n_class1 / (n_class1 + n_class2)
    proportion_class2 = n_class2 / (n_class1 + n_class2)

    n_sample_class1 = int(proportion_class1 * n_samples)
    n_sample_class2 = int(proportion_class2 * n_samples)

    indeces_to_return = []
    for i in range(n_classes):
        idx_class = np.where(ytest == i)[0]
        # idx_class = idx_class.numpy()
        if i == 0:
            idx_for_the_class = np.random.choice(idx_class, n_sample_class1)
        else:
            idx_for_the_class = np.random.choice(idx_class, n_sample_class2)
        print(idx_for_the_class)
        indeces_to_return.extend(idx_for_the_class)
    indeces_to_return = np.array(indeces_to_return)

    return indeces_to_return
