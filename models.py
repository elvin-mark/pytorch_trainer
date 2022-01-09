import torch
import torch.nn as nn


def simple_conv_block(in_planes, out_planes, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2)
    )


def digits_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 16, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(128, 10)
    )


def mnist_cnn():
    return nn.Sequential(
        simple_conv_block(1, 16, 3),
        simple_conv_block(16, 32, 3),
        simple_conv_block(32, 64, 3),
        nn.Flatten(),
        nn.Linear(64, 10)
    )


def cifar10_cnn():
    return nn.Sequential(
        simple_conv_block(3, 16, 3),
        simple_conv_block(16, 32, 3),
        simple_conv_block(32, 64, 3),
        nn.Flatten(),
        nn.Linear(128, 10)
    )


def cifar100_cnn():
    return nn.Sequential(
        simple_conv_block(3, 16, 3),
        simple_conv_block(16, 32, 3),
        simple_conv_block(32, 64, 3),
        nn.Flatten(),
        nn.Linear(128, 100)
    )


def create_model(args):
    if args.model == "digits_cnn":
        return digits_cnn()
    elif args.model == "mnist_cnn":
        return mnist_cnn()
    elif args.model == "cifar10_cnn":
        return cifar10_cnn()
    elif args.model == "cifar100_cnn":
        return cifar100_cnn()
    else:
        return None
