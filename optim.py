import torch


def create_optim(args, model):
    if args.optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        return None
