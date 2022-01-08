import torch
import torch.nn as nn
import torchvision


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
    return test_loss, test_acc


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
    return train_loss, train_acc


def train(model, train_dl, test_dl, crit, optim, epochs, dev, logging=True):
    for epoch in range(epochs):
        train_loss, train_acc = train_one_step(
            model, train_dl, crit, optim, dev)
        test_loss, test_acc = evaluate(model, test_dl, crit, dev)
        if logging:
            print(
                f"train_loss: {train_loss}, train_acc: {train_acc:.2f}%, test_loss: {test_loss}, test_acc: {test_acc:.2f}%")
