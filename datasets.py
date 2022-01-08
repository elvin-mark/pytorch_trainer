import torch
import torch.nn as nn
import torchvision
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def digits_dataloader(args):
    data_ = load_digits()
    X = data_.data.reshape((-1, 1, 8, 8))
    y = data_.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    x_train_tensor = torch.from_numpy(X_train).float()
    x_test_tensor = torch.from_numpy(X_test).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()
    train_ds = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    test_ds = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl


def mnist_dataloader(args):
    train_ds = torchvision.datasets.MNIST(
        "./", train=True, transform=torchvision.transforms.ToTensor())
    test_ds = torchvision.datasets.MNIST(
        "./", train=False, transform=torchvision.transforms.ToTensor())
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl


def create_dataloader(args):
    if args.dataset == "digits":
        return digits_dataloader(args)
    elif args.dataset == "mnist":
        return mnist_dataloader(args)
    else:
        return None
