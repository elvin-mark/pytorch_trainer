import torch
import torch.nn as nn

from parser import create_train_parser
from models import create_model
from datasets import create_dataloader
from optim import create_optim
from utils import train

import os

args = create_train_parser()
if not os.path.exists("trained_models"):
    os.mkdir("trained_models")

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

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

print("Start Training ...")
hist = train(model, train_dl, test_dl, crit, optim,
             args.epochs, dev, logging=args.logging, csv=args.csv, dashboard=args.dashboard, port=args.port, checkpoint=args.checkpoint)

if args.csv and hist is not None:
    hist.to_csv(f"{args.model}_record.csv")

if args.save_model:
    print("Saving model ...")
    torch.save(model.state_dict(), f"trained_models/{args.model}.ckpt")
