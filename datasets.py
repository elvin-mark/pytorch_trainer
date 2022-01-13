import torch
import torch.nn as nn
import torchvision
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pickle
import os

EXTRA_INFO_DIGITS = {"labels": [
    str(i) for i in range(10)], "image_shape": (1, 8, 8)}

EXTRA_INFO_MNIST = {"labels": [
    str(i) for i in range(10)], "image_shape": (1, 28, 28)}

EXTRA_INFO_KMNIST = {"labels": [
    "お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"], "image_shape": (1, 28, 28)}

EXTRA_INFO_FASHION_MNIST = {"labels": [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"], "image_shape": (1, 28, 28)}

EXTRA_INFO_CIFAR10 = {"labels": [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], "image_shape": (3, 32, 32)}

with open("./extra/cifar100_labels.pkl", "rb") as f:
    EXTRA_INFO_CIFAR100 = {"labels": pickle.load(
        f), "image_shape": (3, 32, 32)}

ROOT_DIR = "./data/"


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
    raw_test_ds = test_ds
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)

    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_DIGITS


def mnist_dataloader(args):
    train_ds = torchvision.datasets.MNIST(
        ROOT_DIR, train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_ds = torchvision.datasets.MNIST(
        ROOT_DIR, train=False, download=True, transform=torchvision.transforms.ToTensor())
    raw_test_ds = test_ds
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_MNIST


def kmnist_dataloader(args):
    train_ds = torchvision.datasets.KMNIST(
        ROOT_DIR, train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_ds = torchvision.datasets.KMNIST(
        ROOT_DIR, train=False, download=True, transform=torchvision.transforms.ToTensor())
    raw_test_ds = test_ds
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_KMNIST


def fashion_mnist_dataloader(args):
    train_ds = torchvision.datasets.FashionMNIST(
        ROOT_DIR, train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_ds = torchvision.datasets.FashionMNIST(
        ROOT_DIR, train=False, download=True, transform=torchvision.transforms.ToTensor())
    raw_test_ds = test_ds
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_FASHION_MNIST


def cifar10_dataloader(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [
                                         0.24703233, 0.24348505, 0.26158768])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [
                                         0.24703233, 0.24348505, 0.26158768])
    ])

    train_ds = torchvision.datasets.CIFAR10(
        ROOT_DIR, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(
        ROOT_DIR, train=False, download=True, transform=test_transform)
    raw_test_ds = torchvision.datasets.CIFAR10(
        ROOT_DIR, train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl, test_ds, raw_test_ds, EXTRA_INFO_CIFAR10


def cifar100_dataloader(args):
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [
                                         0.24703233, 0.24348505, 0.26158768])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [
                                         0.24703233, 0.24348505, 0.26158768])
    ])

    train_ds = torchvision.datasets.CIFAR100(
        ROOT_DIR, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR100(
        ROOT_DIR, train=False, download=True, transform=test_transform)
    raw_test_ds = torchvision.datasets.CIFAR100(
        ROOT_DIR, train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size)
    return train_dl, test_dl,  test_ds, raw_test_ds, EXTRA_INFO_CIFAR100


def image_folder_dataloader(args):
    if args.root is None:
        assert("No Root Folder specified")
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((112, 112)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [
                                         0.24703233, 0.24348505, 0.26158768])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((112, 112)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [
                                         0.24703233, 0.24348505, 0.26158768])
    ])
    raw_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((112, 112)),
        torchvision.transforms.ToTensor()
    ])

    train_path = os.path.join(args.root, "train")
    test_path = os.path.join(args.root, "test")
    train_ds = torchvision.datasets.ImageFolder(
        train_path, transform=train_transform)
    test_ds = torchvision.datasets.ImageFolder(
        test_path, transform=test_transform)
    raw_test_ds = torchvision.datasets.ImageFolder(
        test_path, transform=raw_test_transform)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=True)

    extra_info = {"labels": train_ds.classes, "image_shape": (3, 112, 112)}
    return train_dl, test_dl,  test_ds, raw_test_ds, extra_info


def create_dataloader(args):
    if args.dataset == "digits":
        return digits_dataloader(args)
    elif args.dataset == "mnist":
        return mnist_dataloader(args)
    elif args.dataset == "kmnist":
        return kmnist_dataloader(args)
    elif args.dataset == "fashion_mnist":
        return fashion_mnist_dataloader(args)
    elif args.dataset == "cifar10":
        return cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        return cifar100_dataloader(args)
    elif args.dataset == "image_folder":
        return image_folder_dataloader(args)
    else:
        return None
