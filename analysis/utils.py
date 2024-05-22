import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import alexnet, AlexNet_Weights
from enum import Enum
import numpy as np

class Device(Enum):
    CPU = 'cpu'
    GPU0 = 'cuda:0'
    GPU1 = 'cuda:1'
    GPU2 = 'cuda:2'
    GPU3 = 'cuda:3'


class ANET(nn.Module):
    def __init__(self, device):
        super(ANET, self).__init__()
        self.network = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.device = device
        self.network.to(device)

    def pool1(self, x):
        print(self.network.features[:1])
        return self.network.features[:1](x)

    def pool2(self, x):
        # print('feat_shape', x.shape)
        return self.network.features[:3](x)

    def pool3(self, x):
        # print('feat_shape', x.shape)
        return self.network.features[:5](x)

    def pool4(self, x):
        # print('feat_shape', x.shape)
        return self.network.features[:6](x)

    def pool5(self, x):
        # print('feat_shape', x.shape)
        return self.network.features[:7](x)
    
    def fc6(self, x):
        # print('feat_shape', x.shape)
        return self.network.features[:10](x)

    def fc7(self, x):
        # print('feat_shape', x.shape)
        return self.network.features[:12](x)


# # For preprocessing, use torchvision transforms:
# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(image)


def uint8_to_float32(x):
    return x.float() / 255

# Normalize function similar to MXNet version, using PyTorch operations
def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std
