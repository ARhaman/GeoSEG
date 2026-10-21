"""Second segmentation backbone for the backbone-robustness check (script 12).

DeepLabV3+-style network written from scratch (no pretrained weights, so the
comparison with the U-Net is not confounded by ImageNet initialisation):
residual encoder with output stride 8 (dilated last stage), ASPP, and a
low-level-feature decoder. Different inductive bias from the U-Net: atrous
multi-scale context instead of a symmetric skip-connected decoder, bilinear
upsampling instead of transposed convolutions.

Dropout2d (p = C.DROPOUT_P) sits after the last encoder stage, after ASPP and in
the decoder, so MC-Dropout works exactly as for the U-Net (enable_mc_dropout
re-activates every nn.Dropout2d).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def _cbr(cin, cout, k=3, d=1, s=1):
    pad = d * (k - 1) // 2
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, pad, dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, dilation=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, dilation, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, dilation, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)), inplace=True)


class ASPP(nn.Module):
    def __init__(self, cin, cout=128, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([_cbr(cin, cout, 1)] + [_cbr(cin, cout, 3, d=r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False), nn.ReLU(inplace=True))
        self.proj = _cbr(cout * (len(rates) + 2), 256, 1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        g = self.pool(x).expand(-1, -1, x.shape[2], x.shape[3])
        return self.proj(torch.cat(outs + [g], 1))


class DeepLabLite(nn.Module):
    def __init__(self, in_ch=1, n_classes=C.N_CLASSES, w=32, p=C.DROPOUT_P):
        super().__init__()
        self.stem = _cbr(in_ch, w)
        self.l1 = nn.Sequential(ResBlock(w, 2 * w, 2), ResBlock(2 * w, 2 * w))          # 1/2
        self.l2 = nn.Sequential(ResBlock(2 * w, 4 * w, 2), ResBlock(4 * w, 4 * w))      # 1/4
        self.l3 = nn.Sequential(ResBlock(4 * w, 8 * w, 2), ResBlock(8 * w, 8 * w))      # 1/8
        self.l4 = nn.Sequential(ResBlock(8 * w, 8 * w, 1, 2), ResBlock(8 * w, 8 * w, 1, 2))  # 1/8, dilated
        self.drop_enc = nn.Dropout2d(p)
        self.aspp = ASPP(8 * w)
        self.drop_aspp = nn.Dropout2d(p)
        self.low = _cbr(2 * w, 48, 1)
        self.dec = nn.Sequential(_cbr(256 + 48, 128), nn.Dropout2d(p), _cbr(128, 128))
        self.head = nn.Conv2d(128, n_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.stem(x)
        low = self.l1(x)
        x = self.l4(self.l3(self.l2(low)))
        x = self.drop_aspp(self.aspp(self.drop_enc(x)))
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec(torch.cat([x, self.low(low)], 1))
        return F.interpolate(self.head(x), size=(H, W), mode="bilinear", align_corners=False)


BACKBONES = {"deeplab": DeepLabLite}
