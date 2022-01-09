import torch
import torch.nn as nn
import torchvision

import requests
import pandas as pd


class WebLogger:
    def __init__(self, port, token):
        self.url = f"http://localhost:{port}/send_data"

    def send(self, data):
        requests.post(self.url, json=data)


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


def train(model, train_dl, test_dl, crit, optim, epochs, dev, logging=True, csv=False, dashboard=False, token="", port=None):
    data_ = []
    columns = ["epoch", "lr", "train_loss",
               "train_acc", "test_loss", "test_acc"]
    if dashboard:
        web_logger = WebLogger(port, token)

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
            web_logger.send({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                             "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc})
    if csv:
        df = pd.DataFrame(data_, columns=columns)
        return df
