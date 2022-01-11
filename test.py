import torch
import torch.nn as nn

from parser import create_test_parser
from models import create_model
from datasets import create_dataloader
from utils import evaluate, WebLogger, test_images, landscape
import os
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io

args = create_test_parser()

if args.gpu and torch.cuda.is_available():
    print("Using GPU for training")
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")
    print("Using CPU for training. It can be a little bit slow")

model = create_model(args).to(dev)
model.load_state_dict(torch.load(args.model_path, map_location=dev))
train_dl, test_dl,  test_ds, raw_test_ds, extra_info = create_dataloader(args)
crit = nn.CrossEntropyLoss()

print("Start Testing ...")
train_loss, train_acc = evaluate(model, train_dl, crit, dev)
test_loss, test_acc = evaluate(model, test_dl, crit, dev)

print(
    f"train_loss: {train_loss}, train_acc: {train_acc:.2f}%, test_loss: {test_loss}, test_acc: {test_acc:.2f}%")

if args.dashboard:
    print("Preparing Dashboard Logger")
    web_logger = WebLogger(args.port, customize_url=args.url)
    data = {"epoch": 0, "lr": 0, "train_loss": train_loss,
            "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc}
    web_logger.send_data(data)

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

    X, Y, Z = landscape(model, list_models, test_dl, crit,
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
    plt.contour(X, Y, Z)
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.canvas.draw()
    img = Image.frombytes("RGB", fig.canvas.get_width_height(),
                          fig.canvas.tostring_rgb())
    img_mem = io.BytesIO()
    img.save(img_mem, format="PNG")
    img_mem.seek(0)
    img_bytes = img_mem.read()
    img_contour = "data:image/png;base64, " + \
        base64.b64encode(img_bytes).decode()
    results = {"img": img_surf, "contour": img_contour}

    if web_logger is not None:
        web_logger.send_landscape(results)
