import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def test_function(N=50, draw=False):

    alpha = 0.7
    phi_ext = 2 * torch.pi * 0.5

    phi_m = torch.linspace(0, 2 * torch.pi, N)
    phi_p = torch.linspace(0, 2 * torch.pi, N)
    XG, YG = torch.meshgrid(phi_p, phi_m)

    ZG = 2 + alpha - 2 * torch.cos(YG) * torch.cos(XG) - alpha * torch.cos(phi_ext - 2 * YG)

    if draw:
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(XG, YG, ZG, rstride=1, cstride=1, linewidth=0, antialiased=False)
        plt.show()

    X = torch.vstack([XG.ravel(), YG.ravel()]).T
    Y = ZG.ravel()
    return X, Y


def RDB8(N=50, draw=False, normalize=False):
    """
    RDB8 --- Rastrigin function
    """

    A = 10

    x1 = torch.linspace(-5.12, 5.12, N)
    x2 = torch.linspace(-5.12, 5.12, N)
    x1, x2 = torch.meshgrid(x1, x2, indexing=None)
    f = 2 * A + (x1 - A * torch.cos(2 * torch.pi * x1)) + (x2 - A * torch.cos(2 * torch.pi * x2))

    if draw:
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x1, x2, f, rstride=1, cstride=1, linewidth=0, antialiased=False)
        plt.show()

    X = torch.vstack([x1.ravel(), x2.ravel()]).T
    Y = f.ravel()

    if normalize:
        transformer = MinMaxScaler().fit(X)
        X = transformer.transform(X)
        X = torch.tensor(X)

    return X, Y


def RDB7(N=1000, draw=False):
    """
     RDB7
     """

    x = torch.linspace(0, 1, N)

    f = 0.2 * torch.exp(-(10 * x - 4) ** 2) + 0.5 * torch.exp(-(90 * x - 40) ** 2) \
        + 0.3 * torch.exp(-(80 * x - 20) ** 2)

    if draw:
        plt.plot(x, f)
        plt.show()

    return x, f


def MNIST():

    train_dataset = datasets.MNIST(root='data', train=True,
                                   transform=transforms.ToTensor(), download=True)

    Y = train_dataset.targets
    X = train_dataset.data.flatten(1)
    Y = torch.eye(10)[Y]

    return X, Y


def twitter():

    fname = 'data/Twitter.data'

    data = pd.read_csv(fname, header=None)
    data = data.values
    X = data[:, 0:-1]
    Y = data[:, -1]

    transformer = MinMaxScaler().fit(X)
    X = transformer.transform(X)

    transformer = MinMaxScaler().fit(Y.reshape(-1, 1))
    Y = transformer.transform(Y.reshape(-1, 1))

    split = 495763
    X = X[0:split, :]
    Y = Y[0:split]

    return X, Y


if __name__ == "__main__":
    N = 100
    X, Y = RDB8(N, draw=True)
    print(X.shape, Y.shape)

    N = 100
    X, Y = test_function(N, draw=True)
    print(X.shape, Y.shape)

    N = 1000
    X, Y = RDB7(N, draw=True)
    print(X.shape, Y.shape)

    X, Y = CDB1()
    print(X.shape, Y.shape)
