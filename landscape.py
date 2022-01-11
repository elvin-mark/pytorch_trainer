import torch
import torch.nn as nn

from parser import create_landscape_parser
from models import create_model
from datasets import create_dataloader
from utils import landscape
import os
import matplotlib.pyplot as plt

args = create_landscape_parser()

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for training. It can be a little bit slow")

base_model = create_model(args).to(dev)
base_model.load_state_dict(torch.load(args.base_model, map_location=dev))

_, test_dl, _, _, _ = create_dataloader(args)
crit = nn.CrossEntropyLoss()

list_models = [os.path.join(args.checkpoints_dir, elem)
               for elem in os.listdir(args.checkpoints_dir)]

X, Y, Z, proj_checkpoints = landscape(base_model, list_models, test_dl, crit,
                                      args.xrange, args.yrange, args.N, dev)

fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Loss")

plt.figure(2)
plt.contour(X, Y, Z, levels=50)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(proj_checkpoints[:, 0], proj_checkpoints[:, 1])

plt.show()
