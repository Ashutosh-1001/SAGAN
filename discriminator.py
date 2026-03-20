import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_ch: int = 6, ndf: int = 64):
        super().__init__()

        def sn(layer):
            return nn.utils.spectral_norm(layer)

        self.model = nn.Sequential(
            sn(nn.Conv2d(in_ch,   ndf,     4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf,     ndf * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            sn(nn.Conv2d(ndf * 8, 1,       4, stride=1, padding=1)),
        )

    def forward(self, img_in: torch.Tensor, img_target: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([img_in, img_target], dim=1))
