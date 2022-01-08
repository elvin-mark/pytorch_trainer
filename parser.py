from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser(
        description="Simple Trainer for computer vision models")

    parser.add_argument("--dataset", type=str,
                        default="digits", choices=["digits", "mnist"], help="Dataset to be used")
    parser.add_argument("--model", type=str, default="digits_cnn", choices=["digits_cnn", "mnist_cnn"],
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
                        help="Logging training process")
    parser.add_argument("--save-model", action="store_true",
                        help="Save trained model")

    parser.set_defaults(gpu=False, logging=True, save_model=False)
    args = parser.parse_args()
    return args
