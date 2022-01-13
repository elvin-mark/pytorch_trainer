import torch
import torch.nn as nn
from torch.nn.modules import padding


def simple_conv_block(in_planes, out_planes, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2)
    )


class ResBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes,
                               kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            outplanes, outplanes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride > 1 or inplanes != outplanes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o = self.relu2(o + self.shortcut(x))
        return o


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


def digits_resnet():
    return nn.Sequential(
        ResBasicBlock(1, 8, stride=2),
        ResBasicBlock(8, 16),
        ResBasicBlock(16, 32, stride=2),
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


def mnist_resnet():
    return nn.Sequential(
        ResBasicBlock(1, 16, stride=2),
        ResBasicBlock(16, 32, stride=2),
        ResBasicBlock(32, 64, stride=2),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(256, 10)
    )


def cifar10_cnn():
    return nn.Sequential(
        simple_conv_block(3, 16, 3),
        simple_conv_block(16, 32, 3),
        simple_conv_block(32, 64, 3),
        nn.Flatten(),
        nn.Linear(256, 10)
    )


def cifar10_resnet():
    return nn.Sequential(
        ResBasicBlock(3, 16, stride=2),
        ResBasicBlock(16, 32, stride=2),
        ResBasicBlock(32, 64, stride=2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )


def cifar100_cnn():
    return nn.Sequential(
        simple_conv_block(3, 16, 3),
        simple_conv_block(16, 32, 3),
        simple_conv_block(32, 64, 3),
        nn.Flatten(),
        nn.Linear(256, 100)
    )


def simple_general_cnn(num_classes):
    return nn.Sequential(
        simple_conv_block(3, 32, 3),
        nn.MaxPool2d(kernel_size=2),
        simple_conv_block(32, 32, 3),
        nn.MaxPool2d(kernel_size=2),
        simple_conv_block(32, 64, 3),
        nn.MaxPool2d(kernel_size=2),
        simple_conv_block(64, 64, 3),
        nn.MaxPool2d(kernel_size=2),
        simple_conv_block(64, 128, 3),
        nn.MaxPool2d(kernel_size=2),
        simple_conv_block(128, 128, 3),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, num_classes)
    )


def create_model(args):
    if args.model == "digits_cnn":
        return digits_cnn()
    elif args.model == "digits_resnet":
        return digits_resnet()
    elif args.model == "mnist_cnn":
        return mnist_cnn()
    elif args.model == "mnist_resnet":
        return mnist_resnet()
    elif args.model == "cifar10_cnn":
        return cifar10_cnn()
    elif args.model == "cifar10_resnet":
        return cifar10_resnet()
    elif args.model == "cifar100_cnn":
        return cifar100_cnn()
    elif args.model == "simple_general_cnn":
        if args.num_classes is None:
            assert("No number of classes specified. Could not create model")
        return simple_general_cnn(args.num_classes)
    else:
        return None
