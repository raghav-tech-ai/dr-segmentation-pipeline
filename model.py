"""
model.py -- Standard UNet for multi-label lesion segmentation.

Output: 5 channels (MA, HE, EX, SE, OD), sigmoid activation applied in
the loss function (BCEWithLogits) rather than here, for numerical
stability -- so the raw forward() output is logits, not probabilities.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, base_features=32):
        super().__init__()
        f = base_features

        self.enc1 = conv_block(in_channels, f)
        self.enc2 = conv_block(f, f * 2)
        self.enc3 = conv_block(f * 2, f * 4)
        self.enc4 = conv_block(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(f * 8, f * 16)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = conv_block(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = conv_block(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = conv_block(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = conv_block(f * 2, f)

        self.out_conv = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # logits, B x 5 x H x W


if __name__ == "__main__":
    m = UNet(base_features=32)
    x = torch.randn(1, 3, 512, 512)
    y = m(x)
    print("Output shape:", y.shape)  # expect (1, 5, 512, 512)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"Params: {n_params / 1e6:.2f}M")
