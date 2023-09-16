import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    num_epochs: int = 10,
    device: str = "cpu",
):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}")


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
