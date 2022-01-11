import torch
import torch.nn as nn
import torchvision

import requests
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import io
from PIL import Image
import base64


class WebLogger:
    def __init__(self, port, customize_url=None):
        if customize_url:
            self.base_url = customize_url
        else:
            self.base_url = f"http://localhost:{port}/"

    def send_data(self, data):
        requests.post(self.base_url + "send_data", json=data)

    def send_samples(self, data):
        requests.post(self.base_url + "send_samples", json=data)

    def send_landscape(self, data):
        requests.post(self.base_url + "send_landscape", json=data)


def test_images(model, test_ds, raw_test_ds, labels, image_shape, dev, N=5, top=5):
    model.eval()
    softmax = nn.Softmax(dim=1)
    samples = []
    for i, ((x, y), (x_raw, y_raw)) in enumerate(zip(test_ds, raw_test_ds)):
        if i == N:
            break
        if image_shape[0] == 3:
            img = np.round(255 * x_raw.detach().clone().numpy().reshape(
                image_shape).transpose(1, 2, 0)).astype("uint8")
        elif image_shape[0] == 1:
            img = np.round(255*x_raw.detach().clone().numpy().reshape(
                image_shape[1:])).astype("uint8")
        else:
            assert("Invalid Image Shape")
            img = None
        img = Image.fromarray(img)
        img = img.resize((200, 200))
        img_mem = io.BytesIO()
        img.save(img_mem, format="PNG")
        img_mem.seek(0)
        img_bytes = img_mem.read()
        img = "data:image/png;base64, " + \
            base64.b64encode(img_bytes).decode()
        x_ = x.view(1, *image_shape).to(dev)
        o = softmax(model(x_))
        idxs = torch.argsort(o[0], descending=True).cpu()[:top]
        prob = [{"class": labels[i], "prob":o[0][i].item() * 100}
                for i in idxs]
        samples.append({"img": img, "data": prob})
    return {"samples": samples}


def evaluate(model, test_dl, crit, dev):
    model.eval()
    total = 0
    corrects = 0
    tot_loss = 0
    for x, y in test_dl:
        x = x.to(dev)
        y = y.to(dev)
        o = model(x)
        l = crit(o, y)
        corrects += torch.sum(torch.argmax(o, axis=1) == y)
        total += len(y)
        tot_loss += l.item()
    test_loss = tot_loss / len(test_dl)
    test_acc = 100 * corrects / total
    return test_loss, test_acc.item()


def train_one_step(model, train_dl, crit, optim, dev):
    model.train()
    total = 0
    corrects = 0
    tot_loss = 0
    for x, y in train_dl:
        optim.zero_grad()
        x = x.to(dev)
        y = y.to(dev)
        o = model(x)
        l = crit(o, y)
        l.backward()
        optim.step()
        corrects += torch.sum(torch.argmax(o, axis=1) == y)
        total += len(y)
        tot_loss += l.item()
    train_loss = tot_loss / len(train_dl)
    train_acc = 100 * corrects / total
    return train_loss, train_acc.item()


def train(model, train_dl, test_dl, crit, optim, epochs, dev, logging=True, csv=False, dashboard=False, web_logger=None, checkpoint=0):
    data_ = []
    columns = ["epoch", "lr", "train_loss",
               "train_acc", "test_loss", "test_acc"]

    for epoch in range(epochs):
        lr = optim.param_groups[0]["lr"]
        train_loss, train_acc = train_one_step(
            model, train_dl, crit, optim, dev)
        test_loss, test_acc = evaluate(model, test_dl, crit, dev)
        if logging:
            print(
                f"epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc:.2f}%, test_loss: {test_loss}, test_acc: {test_acc:.2f}%")
        data_.append([epoch, lr, train_loss, train_acc, test_loss, test_acc])

        if dashboard:
            web_logger.send_data({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                                  "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})

        if checkpoint > 0 and (epoch) % checkpoint == 0:
            torch.save(model.state_dict(),
                       f"checkpoints/checkpoint_epoch_{epoch}.ckpt")
    if csv:
        df = pd.DataFrame(data_, columns=columns)
        return df


def landscape(base_model, list_models, test_dl, crit, xrange, yrange, N, dev):
    w = [[] for _ in base_model.parameters()]
    base_parameters = [param.detach().clone().reshape(-1)
                       for param in base_model.parameters()]

    for model_path in list_models:
        tmp = torch.load(model_path)
        i = 0
        for k in tmp:
            if "running" in k or "num_batches_tracked" in k:
                continue
            w[i].append(tmp[k].detach().clone().cpu().numpy().reshape(-1))
            i += 1
    w1 = []
    w2 = []
    for w_ in w:
        pca = PCA(n_components=2)
        pca.fit(w_)
        w1.append(torch.from_numpy(pca.components_[0]).float().to(dev))
        w2.append(torch.from_numpy(pca.components_[1]).float().to(dev))

    x_ = np.linspace(*xrange, N)
    y_ = np.linspace(*yrange, N)

    X, Y = np.meshgrid(x_, y_)

    Z = []
    for x, y in zip(X.reshape(-1), Y.reshape(-1)):
        for i, param in enumerate(base_model.parameters()):
            w_ = base_parameters[i] + x * w1[i] + y * w2[i]
            param.data = w_.reshape(param.shape)
        loss, acc = evaluate(base_model, test_dl, crit, dev)
        Z.append(loss)

    Z = np.array(Z).reshape(X.shape)
    return X, Y, Z
