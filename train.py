import torch
import torch.nn as nn
import torchvision

from parser import create_parser
from models import create_model
from datasets import create_dataloader
from optim import create_optim
from utils import train

args = create_parser()

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for training. It can be a little bit slow")

model = create_model(args).to(dev)
train_dl, test_dl = create_dataloader(args)
optim = create_optim(args, model)
crit = nn.CrossEntropyLoss()

print("Starting Training ...")
train(model, train_dl, test_dl, crit, optim,
      args.epochs, dev, logging=args.logging)

if args.save_model:
    print("Saving model ...")
    torch.save(model.state_dict(), f"{args.model}.pth")
