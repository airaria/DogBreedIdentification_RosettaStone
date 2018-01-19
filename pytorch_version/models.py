import torchvision
from torch import nn


def get_resnet50(n_class):
    net = torchvision.models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False
    net.fc = nn.Linear(2048,n_class)
    return net
