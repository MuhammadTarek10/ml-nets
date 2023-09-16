from typing import Tuple, List
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    ToTensor,
    TrivialAugmentWide,
    RandomRotation,
)
from torch.utils.data import Dataset


class CatsDogsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Compose = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.path = list(Path(self.root_dir).glob("**/*.jpg"))
        self.transform = transform
        self.classes, self.classes_to_idx = self.__find_classes()

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        image = self.__load_image(index)
        class_name = self.__get_class(index)
        class_idx = self.classes_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, class_idx

    def __len__(self) -> int:
        return len(self.path)

    def __find_classes(self) -> Tuple[List[str], dict]:
        classes = sorted(
            entry.name for entry in os.scandir(self.root_dir) if entry.is_dir()
        )
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __load_image(self, index: int) -> Image.Image:
        return Image.open(self.path[index]).convert("RGB")

    def __get_class(self, index: int) -> str:
        return self.path[index].parent.name


train_transform = Compose(
    [
        Resize((227, 227)),
        RandomRotation(20),
        TrivialAugmentWide(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = Compose(
    [
        Resize((227, 227)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_cats_dogs = CatsDogsDataset(root_dir="data/train", transform=train_transform)
test_cats_dogs = CatsDogsDataset(root_dir="data/test", transform=test_transform)
