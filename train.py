import io
import torch
import torch.nn as nn
import pickle

from parser import create_train_parser
from models import create_model
from datasets import create_dataloader
from optim import create_optim
from utils import train, WebLogger, test_images, landscape

import matplotlib.pyplot as plt
from PIL import Image
import base64

import os

args = create_train_parser()
if not os.path.exists("trained_models"):
    os.mkdir("trained_models")

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

if not os.path.exists("labels"):
    os.mkdir("labels")

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for training. It can be a little bit slow")

model = create_model(args).to(dev)

if args.start_model is not None:
    try:
        model.load_state_dict(torch.load(args.start_model, map_location=dev))
    except:
        print("Could not load specified model. Using random model")

train_dl, test_dl, test_ds, raw_test_ds, extra_info = create_dataloader(args)
optim = create_optim(args, model)
crit = nn.CrossEntropyLoss()

web_logger = None
if args.dashboard:
    print("Preparing Dashboard Logger")
    web_logger = WebLogger(args.port, customize_url=args.url)

print("Start Training ...")
hist = train(model, train_dl, test_dl, crit, optim,
             args.epochs, dev, logging=args.logging, csv=args.csv, dashboard=args.dashboard, web_logger=web_logger, checkpoint=args.checkpoint)

if args.csv and hist is not None:
    print("Saving CSV Record")
    hist.to_csv(f"{args.model}_record.csv")

if args.save_model:
    print("Saving model ...")
    torch.save(model.state_dict(), f"trained_models/{args.model}.ckpt")

if args.samples:
    print("Generating Sample Images Test")
    results = test_images(
        model, test_ds, raw_test_ds, extra_info["labels"], extra_info["image_shape"], dev)
    if web_logger is not None:
        web_logger.send_samples(results)

if args.landscape:
    print("Generating Landscape")
    list_models = [os.path.join("./checkpoints", elem)
                   for elem in os.listdir("./checkpoints")]

    X, Y, Z, proj_checkpoints = landscape(model, list_models, test_dl, crit,
                                          [-5, 5], [-5, 5], 10, dev)
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Loss")
    fig.canvas.draw()
    img = Image.frombytes("RGB", fig.canvas.get_width_height(),
                          fig.canvas.tostring_rgb())
    img_mem = io.BytesIO()
    img.save(img_mem, format="PNG")
    img_mem.seek(0)
    img_bytes = img_mem.read()
    img_surf = "data:image/png;base64, " + base64.b64encode(img_bytes).decode()

    fig = plt.figure(2)
    plt.contour(X, Y, Z, levels=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(proj_checkpoints[:, 0], proj_checkpoints[:, 1])
    fig.canvas.draw()
    img = Image.frombytes("RGB", fig.canvas.get_width_height(),
                          fig.canvas.tostring_rgb())
    img_mem = io.BytesIO()
    img.save(img_mem, format="PNG")
    img_mem.seek(0)
    img_bytes = img_mem.read()
    img_contour = "data:image/png;base64, " + \
        base64.b64encode(img_bytes).decode()
    results = {"surf": img_surf, "contour": img_contour}

    if web_logger is not None:
        web_logger.send_landscape(results)

if args.labels:
    with open("labels/dataset_labels.pkl", "wb") as f:
        pickle.dump(extra_info["labels"], f)
