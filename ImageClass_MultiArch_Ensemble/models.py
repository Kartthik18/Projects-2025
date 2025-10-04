# models.py
import torch.nn as nn
from torchvision.models import googlenet, resnet18, vgg11

def make_googlenet(num_classes=10, device="cpu"):
    # aux_logits=False keeps the forward simple for CE loss
    model = googlenet(weights=None, aux_logits=False, num_classes=num_classes)
    return model.to(device)

def make_resnet18(num_classes=10, device="cpu"):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def make_vgg11(num_classes=10, device="cpu"):
    model = vgg11(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model.to(device)
