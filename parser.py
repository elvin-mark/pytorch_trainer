from argparse import ArgumentParser

AVAILABLE_DATASETS = ["digits", "mnist", "kmnist",
                      "fashion_mnist", "cifar10", "cifar100", "image_folder"]
AVAILABLE_MODELS = ["digits_cnn", "digits_resnet",
                    "mnist_cnn", "mnist_resnet", "cifar10_cnn", "cifar10_resnet", "cifar100_cnn", "simple_general_cnn", "resnet18"]


def create_train_parser():
    parser = ArgumentParser(
        description="Simple Trainer for computer vision models")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=AVAILABLE_DATASETS, help="Dataset to be used")
    parser.add_argument("--root", type=str, default=None,
                        help="Root folder for images")
    parser.add_argument("--num-classes", type=int,
                        default=None, help="Number of classes")
    parser.add_argument("--model", type=str, default="digits_cnn", choices=AVAILABLE_MODELS,
                        help="Model to be trained")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size to be used in the DataLoaders")
    parser.add_argument("--gpu", action="store_true", help="Train using GPU")
    parser.add_argument("--optim", type=str, default="sgd",
                        help="Optimzer to be used")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--logging", action="store_true",
                        help="Logging training process", dest="logging")
    parser.add_argument("--no-logging", action="store_false",
                        help="No Logging during training", dest="logging")
    parser.add_argument("--save-model", action="store_true",
                        help="Save trained model")
    parser.add_argument("--csv", action="store_true",
                        help="Make a csv file recording the training process")
    parser.add_argument("--dashboard", action="store_true",
                        help="Visualize training process in a dashboard")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port of the dashboard server")
    parser.add_argument("--checkpoint", type=int, default=0,
                        help="Checkpoint frequency")
    parser.add_argument("--landscape", action="store_true",
                        help="Show loss landscape in the dashboard", dest="landscape")
    parser.add_argument("--samples", action="store_true",
                        help="show some sample images in the dashboard", dest="samples")
    parser.add_argument("--url", type=str, default=None,
                        help="Specify a customize URL for the dashboard")
    parser.add_argument("--start-model", type=str, default=None,
                        help="Initial checkpoint for training")
    parser.add_argument("--save-labels", action="store_true",
                        dest="labels", help="Save the labels to map the classes")
    parser.add_argument("--customize", action="store_true",
                        dest="customize", help="Customize models and datasets")
    parser.add_argument("--sched", type=str, default="none",
                        choices=["none", "step"], help="Schedule for learning rate")
    parser.add_argument("--step-size", type=int, default=5,
                        help="Step size for learning rate")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Gamma for learning rate")
    parser.add_argument("--split-dataset", action="store_true",
                        dest="split_dataset", help="split the images in the root folder into train and test datset")
    parser.add_argument("--train-size", type=float, default=0.8,
                        help="train size scale when splitting the dataset into train and test dataset")

    parser.set_defaults(gpu=False, logging=True,
                        save_model=False, csv=False, dashboard=False, landscape=False, samples=False, labels=False, customize=False, split_dataset=False)

    args = parser.parse_args()
    return args


def create_test_parser():
    parser = ArgumentParser(
        description="Simple Tester for computer vision models")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=AVAILABLE_DATASETS, help="Dataset to be used")
    parser.add_argument("--root", type=str, default=None,
                        help="Root folder for images")
    parser.add_argument("--num-classes", type=int,
                        default=None, help="Number of classes")
    parser.add_argument("--model", type=str, default="digits_cnn", choices=AVAILABLE_MODELS,
                        help="Model to be trained")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size to be used in the DataLoaders")
    parser.add_argument("--gpu", action="store_true", help="Train using GPU")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to the saved model")
    parser.add_argument("--landscape", action="store_true",
                        help="Show loss landscape in the dashboard", dest="landscape")
    parser.add_argument("--samples", action="store_true",
                        help="show some sample images in the dashboard", dest="samples")
    parser.add_argument("--url", type=str, default=None,
                        help="Specify a customize URL for the dashboard")
    parser.add_argument("--dashboard", action="store_true",
                        help="Visualize training process in a dashboard")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port of the dashboard server")
    parser.add_argument("--customize", action="store_true",
                        dest="customize", help="Customize models and datasets")

    parser.set_defaults(gpu=False, landscape=False, samples=False)
    args = parser.parse_args()
    return args


def create_landscape_parser():
    parser = ArgumentParser(
        description="Simple program to plot loss landscape")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=AVAILABLE_DATASETS, help="Dataset to be used")
    parser.add_argument("--root", type=str, default=None,
                        help="Root folder for images")
    parser.add_argument("--num-classes", type=int,
                        default=None, help="Number of classes")
    parser.add_argument("--model", type=str, default="digits_cnn", choices=AVAILABLE_MODELS,
                        help="Model to be trained")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size to be used in the DataLoaders")
    parser.add_argument("--gpu", action="store_true", help="Train using GPU")
    parser.add_argument("--base-model", type=str, default="",
                        help="Base model (Best model) path")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints",
                        help="Directory where to find all checkpoints")
    parser.add_argument("--xrange", type=float, nargs="+",
                        default=[-1, 1], help="x range")
    parser.add_argument("--yrange", type=float, nargs="+",
                        default=[-1, 1], help="y range")
    parser.add_argument("--N", type=int, default=10,
                        help="Number of partitions")
    parser.add_argument("--customize", action="store_true",
                        dest="customize", help="Customize models and datasets")

    args = parser.parse_args()
    return args
