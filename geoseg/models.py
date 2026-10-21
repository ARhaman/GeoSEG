"""Models: a dropout-enabled U-Net (the fixed backbone) and Pix2Pix.

Design note: the backbone is deliberately held FIXED across
every arm of Experiment A. The only thing that varies is the data the network
sees. That is what lets us attribute any IoU change to the augmentation
strategy rather than to architecture differences.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(p) if p > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Standard 4-down U-Net. Dropout is placed in the two deepest encoder
    blocks and the bottleneck -- the usual placement for MC-Dropout, since
    dropping shallow features mostly injects texture noise, not model
    uncertainty."""

    def __init__(self, in_ch=1, n_classes=C.N_CLASSES, base=None, p=C.DROPOUT_P):
        base = C.UNET_BASE if base is None else base
        super().__init__()
        self.e1 = ConvBlock(in_ch, base, 0.0)
        self.e2 = ConvBlock(base, base * 2, 0.0)
        self.e3 = ConvBlock(base * 2, base * 4, p)
        self.e4 = ConvBlock(base * 4, base * 8, p)
        self.bott = ConvBlock(base * 8, base * 16, p)
        self.pool = nn.MaxPool2d(2)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = ConvBlock(base * 16, base * 8, p)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = ConvBlock(base * 8, base * 4, p)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 4, base * 2, 0.0)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base * 2, base, 0.0)
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def enable_mc_dropout(model: nn.Module):
    """Put the model in eval mode but re-activate every dropout layer.
    This is the whole trick behind MC-Dropout -- BatchNorm must stay in eval
    (using running stats) or the variance estimate is contaminated by batch
    composition rather than model uncertainty."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            n += 1
    if n == 0:
        raise RuntimeError("no dropout layers found -- MC-Dropout would be a no-op")
    return n


# --------------------------------------------------------------------------
# Pix2Pix  (label mask -> synthetic seismic)
# --------------------------------------------------------------------------
class P2PGenerator(nn.Module):
    """U-Net generator, one-hot facies mask in -> 1-channel seismic out."""

    def __init__(self, n_classes=C.N_CLASSES, base=None):
        super().__init__()
        base = C.P2P_BASE if base is None else base
        self.net = UNet(in_ch=n_classes, n_classes=1, base=base, p=0.0)

    def forward(self, mask_onehot):
        return torch.tanh(self.net(mask_onehot))


class P2PDiscriminator(nn.Module):
    """70x70 PatchGAN on (mask, seismic) pairs."""

    def __init__(self, n_classes=C.N_CLASSES, base=None):
        super().__init__()
        base = C.P2P_BASE if base is None else base
        def blk(cin, cout, stride=2, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *blk(n_classes + 1, base, norm=False),
            *blk(base, base * 2),
            *blk(base * 2, base * 4),
            *blk(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, mask_onehot, seismic):
        return self.net(torch.cat([mask_onehot, seismic], 1))


def one_hot(y, n_classes=C.N_CLASSES):
    return torch.nn.functional.one_hot(y, n_classes).permute(0, 3, 1, 2).float()
