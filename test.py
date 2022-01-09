import torch
import torch.nn as nn

from parser import create_test_parser
from models import create_model
from datasets import create_dataloader
from utils import evaluate

args = create_test_parser()

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for training. It can be a little bit slow")

model = create_model(args).to(dev)
model.load_state_dict(torch.load(args.model_path, map_location=dev))
train_dl, test_dl = create_dataloader(args)
crit = nn.CrossEntropyLoss()

print("Start Testing ...")
train_loss, train_acc = evaluate(model, train_dl, crit, dev)
test_loss, test_acc = evaluate(model, test_dl, crit, dev)

print(
    f"train_loss: {train_loss}, train_acc: {train_acc:.2f}%, test_loss: {test_loss}, test_acc: {test_acc:.2f}%")
