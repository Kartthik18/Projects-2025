# main.py
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loaders
from models import make_googlenet, make_resnet18, make_vgg11
from train import train_model, test_ensemble
from utils_vis import preview_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    trainloader, testloader, classes = get_loaders(batch_size_train=128, batch_size_test=100, img_size=96)

    # Models
    model_googlenet = make_googlenet(num_classes=10, device=device)
    model_resnet    = make_resnet18(num_classes=10, device=device)
    model_vgg       = make_vgg11(num_classes=10, device=device)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Train each model (same LR as your script)
    train_model(model_googlenet, trainloader,
                optim.Adam(model_googlenet.parameters(), lr=1e-3),
                criterion, device, num_epochs=10)

    train_model(model_resnet, trainloader,
                optim.Adam(model_resnet.parameters(), lr=1e-3),
                criterion, device, num_epochs=10)

    train_model(model_vgg, trainloader,
                optim.Adam(model_vgg.parameters(), lr=1e-3),
                criterion, device, num_epochs=10)

    # Ensemble test
    test_ensemble([model_googlenet, model_resnet, model_vgg], testloader, device)

    # Preview a few predictions from a test batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    preview_batch([model_googlenet, model_resnet, model_vgg], images, labels, classes, device)

if __name__ == "__main__":
    main()
