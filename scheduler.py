import torch


def create_lr_scheduler(args, optim):
    if args.sched == "step":
        return torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=args.gamma)
    return None
