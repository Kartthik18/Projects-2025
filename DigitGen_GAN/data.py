# data.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size: int = 128, img_size: int = 28, root: str = "./data"):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # maps to [-1, 1]
    ])
    dataset = datasets.MNIST(root=root, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
