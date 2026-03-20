import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class LSGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def discriminator(self, real_pred, fake_pred):
        return 0.5 * (self.mse(real_pred, torch.ones_like(real_pred)) +
                      self.mse(fake_pred, torch.zeros_like(fake_pred)))

    def generator(self, fake_pred):
        return self.mse(fake_pred, torch.ones_like(fake_pred))


class VGGPerceptualLoss(nn.Module):
    VGG_LAYERS = [3, 8, 17]

    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for end in self.VGG_LAYERS:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end+1]))
            prev = end + 1
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def _norm(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, pred, target):
        pred, target = self._norm(pred), self._norm(target)
        loss = 0.0
        for s in self.slices:
            pred = s(pred)
            target = s(target)
            loss += F.l1_loss(pred, target.detach())
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1,1,3,3).repeat(3,1,1,1))
        self.register_buffer("ky", ky.view(1,1,3,3).repeat(3,1,1,1))

    def _sobel(self, x):
        gx = F.conv2d(x, self.kx, padding=1, groups=3)
        gy = F.conv2d(x, self.ky, padding=1, groups=3)
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    def forward(self, pred, target):
        return F.l1_loss(self._sobel(pred), self._sobel(target))
