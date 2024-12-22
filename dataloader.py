from torchvision.datasets import CIFAR10
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Albumentations transforms
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                    fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=None, p=0.5),
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ToTensorV2(),
])

# Custom Dataset
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

# Load CIFAR-10 dataset
def get_dataloaders(batch_size=128, num_workers=4, seed=1):
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)

    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=None)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=None)

    train_dataset = AlbumentationsDataset(train_dataset, transform=train_transforms)
    test_dataset = AlbumentationsDataset(test_dataset, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    return train_loader, test_loader
