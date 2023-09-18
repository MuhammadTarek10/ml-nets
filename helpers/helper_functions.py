from typing import Tuple, List
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import Accuracy
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_step(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    running_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    return train_loss


def test_step(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    return test_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    num_epochs: int = 10,
    device: str = "cpu",
) -> Tuple[List[int], List[float], List[float]]:
    epochs_count, train_losses, test_losses = [], [], []
    for epoch in tqdm(range(num_epochs)):
        epochs_count.append(epoch)
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        test_loss = test_step(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        tqdm.write(
            f"Epoch {epoch + 1}: train loss {train_loss:0.3f} test loss {test_loss:0.3f}"
        )
    return epochs_count, train_losses, test_losses


def predict(
    model: nn.Module,
    image: torch.Tensor,
    device: str = "cpu",
):
    model.eval()
    with torch.inference_mode():
        output = model(image.to(device))
        return output
        # _, predicted = torch.max(output.data, 1)
        # return predicted.item()


def display_random(dataset, classes, model=None, n=10, device="cpu"):
    if model:
        model.to(device)
    fig, ax = plt.subplots(n // 5, n // 2, figsize=(15, 8))
    idx = random.sample(range(len(dataset)), k=n)
    counter = 0
    for i in range(n // 5):
        for j in range(n // 2):
            image, label = dataset[idx[counter]]
            image = image.to(device)
            if model:
                preds = predict(model, image.unsqueeze(dim=0))
            image = image.permute(1, 2, 0)
            ax[i, j].imshow(image)
            if model:
                color = "green" if preds == label else "red"
                ax[i, j].set_title(
                    f"Actual: {classes[label]} | Predicted: {classes[preds]}",
                    fontsize=8,
                    color=color,
                )
            else:
                ax[i, j].set_title(f"Actual: {classes[label]}", fontsize=22)
            counter += 1


def plot_stats(
    epochs: list,
    train_loss: list,
    test_loss: list,
):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accs(
    epochs: list,
    accs: list,
):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [i.cpu() for i in accs], label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def accuracy(model, data_loader, device="cpu"):
    acc = Accuracy(task="binary", num_classes=2)
    model.to(device)
    model.eval()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        acc.update(output, y)
    return acc.compute()
