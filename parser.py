from argparse import ArgumentParser

AVAILABLE_DATASETS = ["digits", "mnist", "cifar10", "cifar100"]
AVAILABLE_MODELS = ["digits_cnn", "mnist_cnn", "cifar10_cnn", "cifar100_cnn"]


def create_train_parser():
    parser = ArgumentParser(
        description="Simple Trainer for computer vision models")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=AVAILABLE_DATASETS, help="Dataset to be used")
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

    parser.set_defaults(gpu=False, logging=True,
                        save_model=False, csv=False, dashboard=False)

    args = parser.parse_args()
    return args


def create_test_parser():
    parser = ArgumentParser(
        description="Simple Tester for computer vision models")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=AVAILABLE_DATASETS, help="Dataset to be used")
    parser.add_argument("--model", type=str, default="digits_cnn", choices=AVAILABLE_MODELS,
                        help="Model to be trained")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size to be used in the DataLoaders")
    parser.add_argument("--gpu", action="store_true", help="Train using GPU")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to the saved model")

    parser.set_defaults(gpu=False)
    args = parser.parse_args()
    return args


def create_landscape_parser():
    parser = ArgumentParser(
        description="Simple program to plot loss landscape")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=AVAILABLE_DATASETS, help="Dataset to be used")
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

    args = parser.parse_args()
    return args
