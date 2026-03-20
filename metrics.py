import torch
import torch.nn.functional as F


def _to_01(x):
    return (x.clamp(-1, 1) + 1) / 2


def psnr(pred, target, max_val=1.0):
    pred, target = _to_01(pred), _to_01(target)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1,2,3))
    return (10 * torch.log10(max_val**2 / (mse + 1e-8))).mean().item()


def _gaussian_kernel(size=11, sigma=1.5, channels=3):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1)


def ssim(pred, target, window_size=11, sigma=1.5, K1=0.01, K2=0.03):
    pred, target = _to_01(pred), _to_01(target)
    C = pred.size(1)
    k = _gaussian_kernel(window_size, sigma, C).to(pred.device)
    p = window_size // 2
    mu_p = F.conv2d(pred, k, padding=p, groups=C)
    mu_t = F.conv2d(target, k, padding=p, groups=C)
    sig_p = F.conv2d(pred**2, k, padding=p, groups=C) - mu_p**2
    sig_t = F.conv2d(target**2, k, padding=p, groups=C) - mu_t**2
    sig_pt = F.conv2d(pred*target, k, padding=p, groups=C) - mu_p*mu_t
    C1, C2 = K1**2, K2**2
    num = (2*mu_p*mu_t + C1) * (2*sig_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return (num / (den + 1e-8)).mean().item()


class MetricTracker:
    def __init__(self): self.reset()
    def reset(self): self._sp = self._ss = self._n = 0.0
    def update(self, pred, target):
        self._sp += psnr(pred, target)
        self._ss += ssim(pred, target)
        self._n += 1
    @property
    def avg_psnr(self): return self._sp / max(self._n, 1)
    @property
    def avg_ssim(self): return self._ss / max(self._n, 1)
    def summary(self): return {"PSNR": self.avg_psnr, "SSIM": self.avg_ssim}
