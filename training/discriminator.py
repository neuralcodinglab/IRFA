import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

class History(nn.Module):
    def __init__(self, capacity: int):
        super(History, self).__init__()
        self._capacity = capacity
        self._data = []

    def __call__(self, z_prime: torch.Tensor) -> torch.Tensor:
        z = []
        for i in range(z_prime.size(0)):
            if len(self._data) < self._capacity:
                # Detach tensor before adding to history to prevent unwanted gradient computations
                z.append(z_prime[i].detach())
                self._data.append(z_prime[i].detach())
            elif torch.rand(1) < 0.5:
                index = torch.randint(0, self._capacity, (1,)).item()
                selected = self._data.pop(index)
                z.append(selected)
                self._data.append(z_prime[i].detach())  # Ensure detached tensor is added
            else:
                z.append(z_prime[i].detach())  # Ensure detached tensor is added
        return torch.stack(z, dim=0)

    
class Lossfun(nn.Module):
    def __init__(self, alpha: float):
        super(Lossfun, self).__init__()
        self._alpha = alpha
        self._bce = nn.BCEWithLogitsLoss()

    def __call__(self, p: float, p_hat: torch.Tensor) -> torch.Tensor:
        target = torch.full(p_hat.size(), p, device=p_hat.device, dtype=torch.float)
        return self._alpha * self._bce(p_hat, target)

class Network(nn.Module):
    def __init__(self, count: int, depth: int):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(depth, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, count, 3, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self._history = History(50)
        self._lossfun = Lossfun(1)
        self._network = Network(1, input_channels + 3)
        
        self._trainer = optim.Adam(self._network.parameters(), lr=0.0001, betas=(0.3, 0.999))
    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun
        
    @property
    def history(self) -> History:
        return self._history

    @property
    def network(self) -> nn.Module:
        return self._network

    def train(self, g: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        '''
        g: generator instance
        x: input to generator
        y: targets
        '''
        
        self._network.train()
        z = self._history(torch.cat([x, g(x)], dim=1))

        loss = 0
        self._trainer.zero_grad()
        with torch.set_grad_enabled(True):
            
            loss += 0.5 * (self.lossfun(float(0), self.network(z)) + self.lossfun(float(1), self.network(torch.cat((x, y), dim=1))))

            loss.backward(retain_graph=True)

        self._trainer.step()

        return float(loss.item())
