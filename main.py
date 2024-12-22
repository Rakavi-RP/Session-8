import torch
import torch.optim as optim
from model import CIFAR10
from train import train, test
from dataloader import get_dataloaders
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load data
train_loader, test_loader = get_dataloaders()

# Define model, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=3, verbose=True)

# Training loop
EPOCHS = 150
for epoch in range(1, EPOCHS + 1):
    print(f"EPOCH: {epoch}")
    train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader)
    scheduler.step(test_loss)
