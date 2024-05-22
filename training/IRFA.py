import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import models, transforms
from typing import Tuple
from typing import Callable, Tuple

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        self.vgg19.eval()  # Set to eval mode
        self.transformer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.l1_loss = nn.L1Loss()

    def forward(self, y_hat, y):
        
        y_hat = F.interpolate(y_hat, size=(224, 224), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(224, 224), mode='bilinear', align_corners=False)
        
        y_hat = self.transformer(y_hat)
        y = self.transformer(y)

        feat_layers = [8, 15, 22, 29, 35]
        loss = 0
        for layer in feat_layers:
            y_feat = self.vgg19[:layer](y)
            y_hat_feat = self.vgg19[:layer](y_hat)
            loss += self.l1_loss(y_hat_feat, y_feat)
            
        return loss / len(feat_layers)

class Lossfun(nn.Module):
    def __init__(self, alpha, beta_vgg, beta_pix):
        super(Lossfun, self).__init__()
        self._alpha = alpha
        self._beta_vgg = beta_vgg
        self._beta_pix = beta_pix
        self._bce = nn.BCEWithLogitsLoss()
        self._l1 = nn.L1Loss()
        self._vgg_loss = VggLoss()

    def forward(self, p, p_hat, y, y_hat): 
        '''
        # ones, d_out, target, g_out
        
        '''
        dis_loss = self._alpha * self._bce(p_hat, torch.ones_like(p_hat))
        gen_loss_vgg = self._beta_vgg * self._vgg_loss(y_hat, y)
        gen_loss_pix = self._beta_pix * self._l1(y_hat, y)
        total_loss = dis_loss + gen_loss_vgg + gen_loss_pix
        return total_loss, dis_loss, gen_loss_vgg, gen_loss_pix
    
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def l1(self):
        return self._l1
        
        
class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    @property
    def count(self):
        raise NotImplementedError

    @property
    def depth(self):
        raise NotImplementedError
        
class Identity(Layer):
    def __init__(self, count, depth):
        super(Identity, self).__init__()
        self._count = count
        self._depth = depth

    @property
    def count(self):
        return self._count

    @property
    def depth(self):
        return self._depth

    def forward(self, x):
        return x
        
class Skip(Layer):
    def __init__(self, count, depth, layer):
        super(Skip, self).__init__()
        self._layer = layer
        self._count = count
        self._depth = depth

        self.block = nn.Sequential(
            nn.Conv2d(depth, layer.depth, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(layer.depth),
            nn.LeakyReLU(0.2, inplace=True),
            layer,
            nn.ConvTranspose2d(layer.count, count, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(count)
        )
        
    @property
    def count(self) -> int:
        return self._count + self._depth

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def layer(self) -> Layer:
        return self._layer

    def forward(self, x):
        return F.relu(torch.cat([x, self.block(x)], dim=1))


class InverseReceptiveField(nn.Module):
    """
    Inverse receptive field module.

    Attributes:
        M (MultiheadEmbedding): Multihead embedding module.
        P (PositionalEncoding): Positional encoding module.
        T (TransposeAttention): Transpose attention module.

    Args:
        h (int): Number of heads.
        c (int): Number of channels.
        i (int): Number of image pixels.
        k (int): Number of kernel pixels.
    """
    def __init__(self, h: int, c: int, i: int, k: int) -> None:
        super().__init__()
        self.m = MultiheadEmbedding(h, c)
        self.p = PositionalEncoding(c, i)
        self.t = TransposeAttention(h, c, k)

    def forward(self, X: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (Tuple[torch.Tensor, ...]): Input matrices (h, (n, *)).

        Returns:
            torch.Tensor: Transpose attention tensor (n, c, i, i).
        """
        _M=self.m(X)        
        _P = self.p(_M)
        _T, _w = self.t(_P)
        return _T, _M, _P, _w
        
class MultiheadEmbedding(nn.Module):
    """
    Multihead embedding module with nonlinear layers.
    
    Attributes:
        M (nn.ModuleList): List of embedding sequences for each head.

    Args:
        h (int): Number of heads.
        c (int): Number of channels.
    """
    def __init__(self, h: int, c: int, activation_fn=nn.ReLU) -> None:
        super().__init__()
        self.M = nn.ModuleList([nn.Sequential(
            nn.LazyLinear(c // 2),  # or nn.Linear(input_dim, c // 2) if input dimension is known
            nn.BatchNorm1d(c // 2),
            activation_fn()
        ) for _ in range(h)])

    def forward(self, X: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X (Tuple[torch.Tensor, ...]): Input matrices (h, (n, *)).

        Returns:
            torch.Tensor: Multihead embedding tensor (n, h, c // 2, 1, 1).
        """
        X = [m(x) for m, x in zip(self.M, X)]
        return torch.stack(X, dim=1)[..., None, None]

class PositionalEncoding(nn.Module):
    """
    Positional encoding module.

    Attributes:
        P (nn.ParameterDict): Positional encoding translations.

    Args:
        c (int): Number of channels.
        i (int): Number of image pixels.
    """
    def __init__(self, c: int, i: int) -> None:
        super().__init__()
        self.P = nn.ParameterDict({
            'x': nn.Parameter(torch.empty(1, 1, c // 2, 1, i).normal_(std=c ** -0.5)),
            'y': nn.Parameter(torch.empty(1, 1, c // 2, i, 1).normal_(std=c ** -0.5))
        })


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            m (torch.Tensor): Multihead embedding tensor (n, h, c // 2, 1, 1).

        Returns:
            torch.Tensor: Positional encoding tensor (n, h, c, i, i).
        """
        return torch.cat((
            (x + self.P['x']).expand(-1, -1, -1, self.P['y'].size(3), -1),
            (x + self.P['y']).expand(-1, -1, -1, -1, self.P['x'].size(4))
        ), 2)


class TransposeAttention(nn.Module):
    """
    Transpose attention module.

    Attributes:
        k (nn.Conv2d): Transpose attention key convolution.
        v (nn.Conv2d): Transpose attention value convolution.
        s (nn.Sequential): Transpose attention score convolution.

    Args:
        h (int): Number of heads.
        c (int): Number of channels.
        k (int): Number of kernel pixels.
    """
    def __init__(self, h: int, c: int, k: int) -> None:
        super().__init__()
        self._w: Optional[torch.Tensor] = None        
        self.k = nn.Conv2d(h * c, h * c, 1, groups=h)
        self.v = nn.Conv2d(h * c, h * c, 1, groups=h)
        self.s = nn.Sequential(nn.Conv2d(c, c, k, padding='same'), _Lambda(lambda d: d / (c * k * k) ** 0.5))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            p (torch.Tensor): Positional encoding tensor (n, h, c, i, i).

        Returns:
            torch.Tensor: Transpose attention tensor (n, c, i, i).
        """
        w = F.softmax(self.s(self.k(p.flatten(1, 2)).view(-1, *p.size()[2:])).view(*p.size()), 1)
        self._w = w.detach()
        
        return (w * self.v(p.flatten(1, 2)).view(*p.size())).sum(1), self._w

class _Lambda(nn.Module):
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self._f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._f(x)
        
class Generator(nn.Module):
    def __init__(self, h: int, c: int, i: int, k: int, alpha=0.01, beta_vgg=1, beta_pix=0.5, lr = 0.0002):
        super(Generator, self).__init__()
        self._lossfun = Lossfun(alpha=float(alpha), beta_vgg=float(beta_vgg), beta_pix=float(beta_pix))
        self._irfa = InverseReceptiveField(h, c, i, k)
        self._network = Network(count=3, conv_depth=c)
        self._network.apply(init_weights)
        params = list(self._irfa.parameters()) + list(self._network.parameters())
        self.optimizer = Adam(params, lr=lr, betas=(0.5, 0.999))
        
    @property
    def irfa(self) -> nn.Module:
        return self._irfa
        
    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun

    @property
    def network(self) -> nn.Module:
        return self._network
        
    def train(self, d, x, y):

        self._network.train()
        self._irfa.train() 
       
        irfa_map, M, P, _w = self._irfa(x)
        total_loss, dis_loss, gen_loss_vgg, gen_loss_pix = (lambda y_hat: self._lossfun(
            torch.tensor(1), 
            d(torch.cat([irfa_map, y_hat], dim=1)), 
            y, 
            y_hat))(self._network(irfa_map))
    
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            total_loss.backward()

        self.optimizer.step()

        return total_loss.item(), dis_loss.item(), gen_loss_vgg.item(), gen_loss_pix.item()

class Network(nn.Module):
    def __init__(self, count: int, conv_depth: int):
        super(Network, self).__init__()

        self._count = count
        self._depth = conv_depth
        
        modules = [
            nn.Conv2d(conv_depth, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        layer = Identity(512, 512)  
        layer = Skip(512, 512, layer)  
        layer = Skip(256, 256, layer)
        layer = Skip(128, 128, layer)
        layer = Skip(64, 64, layer)

        modules.append(layer)
        
        modules += [
            nn.ConvTranspose2d(128, count, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ]
        
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)

    @property
    def count(self):
        return self._count

    @property
    def depth(self):
        return self._depth





