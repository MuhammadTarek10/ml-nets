import torch
from torch import nn

from ..dataset import test_cats_dogs
from ..helpers.helper_functions import predict


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=2),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        x = x.unsqueeze(dim=0) if x.dim() != 4 else x
        x = x.view(x.size(0), -1)
        return self.classification_head(x)


if __name__ == "__main__":
    image, labels = next(iter(test_cats_dogs))
    model = AlexNet()
    print(f"Image Shape: {image.shape}")
    print(f"Prediction: {predict(model, image)}")
