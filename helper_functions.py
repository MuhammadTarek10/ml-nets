from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm


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
