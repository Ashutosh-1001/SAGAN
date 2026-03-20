import torch
import torch.nn as nn
from .attention import CBAM


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_norm=True, dropout=0.0):
        super().__init__()
        if down:
            conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm)
        else:
            conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm)
        layers = [conv]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
        self.act_down = nn.LeakyReLU(0.2, inplace=True)
        self.act_up = nn.ReLU(inplace=True)
        self.down = down

    def forward(self, x):
        x = self.block(x)
        return self.act_down(x) if self.down else self.act_up(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, use_attention=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )
        self.attn = CBAM(ch) if use_attention else nn.Identity()

    def forward(self, x):
        return x + self.attn(self.block(x))


class Generator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 ngf: int = 64, n_residual: int = 9):
        super().__init__()

        self.e1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        )
        self.e2 = ConvBnRelu(ngf,     ngf * 2)
        self.e3 = ConvBnRelu(ngf * 2, ngf * 4)
        self.e4 = ConvBnRelu(ngf * 4, ngf * 8)
        self.e5 = ConvBnRelu(ngf * 8, ngf * 8)
        self.e6 = ConvBnRelu(ngf * 8, ngf * 8)
        self.e7 = ConvBnRelu(ngf * 8, ngf * 8)
        self.e8 = ConvBnRelu(ngf * 8, ngf * 8, use_norm=False)

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(ngf * 8, use_attention=True) for _ in range(n_residual)]
        )

        self.d1 = ConvBnRelu(ngf * 8,     ngf * 8, down=False, dropout=0.5)
        self.d2 = ConvBnRelu(ngf * 8 * 2, ngf * 8, down=False, dropout=0.5)
        self.d3 = ConvBnRelu(ngf * 8 * 2, ngf * 8, down=False, dropout=0.5)
        self.d4 = ConvBnRelu(ngf * 8 * 2, ngf * 8, down=False)
        self.d5 = ConvBnRelu(ngf * 8 * 2, ngf * 4, down=False)
        self.d6 = ConvBnRelu(ngf * 4 * 2, ngf * 2, down=False)
        self.d7 = ConvBnRelu(ngf * 2 * 2, ngf,     down=False)

        self.skip_attn = nn.ModuleList([
            CBAM(ngf * 8),
            CBAM(ngf * 8),
            CBAM(ngf * 8),
            CBAM(ngf * 8),
            CBAM(ngf * 4),
            CBAM(ngf * 2),
            CBAM(ngf),
        ])

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf * 2, out_ch, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        s5 = self.e5(s4)
        s6 = self.e6(s5)
        s7 = self.e7(s6)
        bn = self.e8(s7)
        bn = self.bottleneck(bn)

        d = self.d1(bn)
        d = self.d2(torch.cat([d, self.skip_attn[0](s7)], 1))
        d = self.d3(torch.cat([d, self.skip_attn[1](s6)], 1))
        d = self.d4(torch.cat([d, self.skip_attn[2](s5)], 1))
        d = self.d5(torch.cat([d, self.skip_attn[3](s4)], 1))
        d = self.d6(torch.cat([d, self.skip_attn[4](s3)], 1))
        d = self.d7(torch.cat([d, self.skip_attn[5](s2)], 1))
        return self.out(torch.cat([d, self.skip_attn[6](s1)], 1))
