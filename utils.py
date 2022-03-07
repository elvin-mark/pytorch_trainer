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
import tqdm
import matplotlib.pyplot as plt
import networkx as nx


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
    random_idxs = np.random.randint(0, len(test_ds), (N,))
    for i in random_idxs:
        x, y = test_ds[i]
        x_raw, y_raw = raw_test_ds[i]
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
        with torch.no_grad():
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


def train(model, train_dl, test_dl, crit, optim, epochs, dev, lr_sched=None, logging=True, csv=False, dashboard=False, web_logger=None, checkpoint=0):
    data_ = []
    columns = ["epoch", "lr", "train_loss",
               "train_acc", "test_loss", "test_acc"]

    for epoch in tqdm.tqdm(range(epochs)):
        lr = optim.param_groups[0]["lr"]
        train_loss, train_acc = train_one_step(
            model, train_dl, crit, optim, dev)
        test_loss, test_acc = evaluate(model, test_dl, crit, dev)
        if lr_sched:
            lr_sched.step()
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

    ###########################################################
    proj_checkpoints = []

    for model_path in list_models:
        tmp = torch.load(model_path)
        i = 0
        acc_proj1 = 0
        acc_norm1 = 0
        acc_proj2 = 0
        acc_norm2 = 0

        for k in tmp:
            if "running" in k or "num_batches_tracked" in k:
                continue
            acc_proj1 += w1[i].dot(tmp[k].reshape(-1) -
                                   base_parameters[i]).item()
            acc_norm1 += w1[i].dot(w1[i]).item()
            acc_proj2 += w2[i].dot(tmp[k].reshape(-1) -
                                   base_parameters[i]).item()
            acc_norm2 += w2[i].dot(w2[i]).item()
            i += 1
        proj_checkpoints.append(
            [acc_proj1 / acc_norm1 ** 0.5, acc_proj2 / acc_norm2 ** 0.5])
    proj_checkpoints = np.array(proj_checkpoints)
    ###########################################################

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
    return X, Y, Z, proj_checkpoints


def get_model_graph(model, dummy, dev):
    """
    Function that returns a graph of the operations done in a neural network model.
    """
    nodes_dict = {}
    layers_dict = {}

    class MyTuple(tuple):
        """
        Tuple class for registiring sum between tensors during forward operation.
        """
        sum_idx = 0

        def __init__(self, x):
            self.raw = x

        def __add__(self, y):
            i, x_ = self.raw
            j, y_ = y.raw
            MyTuple.sum_idx += 1
            sum_node = f'sum_{MyTuple.sum_idx}'
            nodes_dict[sum_node] = {"from": [i, j]}

            return (sum_node, x_+y_)

    def prehook_fn(self, input):
        """
        Preforward hook: register the connection between the previous layer and the current layer
        """
        tmp = str(id(self))
        if tmp not in nodes_dict:
            nodes_dict[tmp] = {"from": []}
        if tmp not in layers_dict:
            layers_dict[tmp] = str(type(self)).split(".")[-1][:-2]
        ind, input_ = input[0]
        nodes_dict[tmp]["from"].append(ind)
        return input_

    def hook_fn(self, input, output):
        """
        Forward hook: returns a tuple with the id of the current layer and the output of the current layer
        """
        tmp = str(id(self))
        return MyTuple((tmp, output))

    def register_hooks(model):
        """
        Register recursively to all layers the forward hook
        """
        if "torch" not in str(type(model)) or "Sequential" in str(type(model)):
            for child in model.children():
                register_hooks(child)

        else:
            model.register_forward_hook(hook_fn)

    def register_pre_hooks(model):
        """
        Register recursively to all layers the pre forward hook
        """
        if "torch" not in str(type(model)) or "Sequential" in str(type(model)):
            for child in model.children():
                register_pre_hooks(child)

        else:
            model.register_forward_pre_hook(prehook_fn)

    register_hooks(model)
    register_pre_hooks(model)

    # Forward the dummy variable
    model(("init", dummy.to(dev)))

    layers_dict["init"] = "init"

    for i in range(1, MyTuple.sum_idx+1):
        layers_dict[f"sum_{i}"] = f"sum_{i}"

    return nodes_dict, layers_dict


def draw_graph(nodes_dict, layers_dict):
    edges = []

    for k, v in nodes_dict.items():
        for t in v["from"]:
            edges.append([t, k])

    g = nx.Graph()
    g.add_edges_from(edges)
    nx.draw_networkx(g, labels=layers_dict)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = buf.read()
    img = "data:image/png;base64, " + \
        base64.b64encode(img_bytes).decode()

    return img
