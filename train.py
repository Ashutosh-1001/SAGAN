import argparse, random, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from models import Generator, Discriminator
from data import get_dataloader
from utils import LSGANLoss, VGGPerceptualLoss, EdgeLoss, MetricTracker
from utils.visualize import save_comparison_grid


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_lr_lambda(total_epochs):
    decay_start = total_epochs // 2
    def lr_lambda(epoch):
        if epoch < decay_start: return 1.0
        return max(1.0 - (epoch - decay_start) / max(total_epochs - decay_start, 1), 0.0)
    return lr_lambda


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_one_step(batch, G, D, opt_G, opt_D, gan_loss, perc_loss,
                   edge_fn, scaler_G, scaler_D, device, cfg):
    cloudy = batch["cloudy"].to(device, non_blocking=True)
    clear  = batch["clear"].to(device, non_blocking=True)

    opt_D.zero_grad(set_to_none=True)
    with autocast(enabled=cfg.amp):
        fake   = G(cloudy).detach()
        loss_D = gan_loss.discriminator(D(cloudy, clear), D(cloudy, fake))
    scaler_D.scale(loss_D).backward()
    scaler_D.unscale_(opt_D)
    nn.utils.clip_grad_norm_(D.parameters(), cfg.clip_grad)
    scaler_D.step(opt_D); scaler_D.update()

    opt_G.zero_grad(set_to_none=True)
    with autocast(enabled=cfg.amp):
        fake   = G(cloudy)
        l_gan  = gan_loss.generator(D(cloudy, fake))
        l_pix  = nn.functional.l1_loss(fake, clear)
        l_perc = perc_loss(fake, clear)
        l_edge = edge_fn(fake, clear)
        loss_G = (cfg.lambda_gan   * l_gan  + cfg.lambda_pixel * l_pix +
                  cfg.lambda_perc  * l_perc + cfg.lambda_edge  * l_edge)
    scaler_G.scale(loss_G).backward()
    scaler_G.unscale_(opt_G)
    nn.utils.clip_grad_norm_(G.parameters(), cfg.clip_grad)
    scaler_G.step(opt_G); scaler_G.update()

    return {"loss/D": loss_D.item(), "loss/G_total": loss_G.item(),
            "loss/G_gan": l_gan.item(), "loss/G_pixel": l_pix.item(),
            "loss/G_perc": l_perc.item(), "loss/G_edge": l_edge.item()}


@torch.no_grad()
def validate(G, loader, device, cfg, epoch, sample_dir):
    G.eval(); tracker = MetricTracker(); saved = False
    for batch in loader:
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear  = batch["clear"].to(device, non_blocking=True)
        fake   = G(cloudy)
        tracker.update(fake, clear)
        if not saved:
            save_comparison_grid(cloudy, fake, clear,
                                 str(sample_dir / f"epoch_{epoch:04d}.png"))
            saved = True
    G.train()
    return tracker.summary()


def main(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    print(f"[device] {device}")

    train_loader = get_dataloader(cfg.data_root, "train", cfg.image_size,
                                  cfg.batch_size, cfg.num_workers)
    val_loader   = get_dataloader(cfg.data_root, "val",   cfg.image_size,
                                  cfg.batch_size, cfg.num_workers)

    G = Generator(ngf=cfg.ngf, n_residual=cfg.n_residual).to(device)
    D = Discriminator(ndf=cfg.ndf).to(device)

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        G.load_state_dict(ckpt["G"]); D.load_state_dict(ckpt["D"])

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, build_lr_lambda(cfg.epochs))
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, build_lr_lambda(cfg.epochs))

    gan_loss  = LSGANLoss()
    perc_loss = VGGPerceptualLoss(device).to(device)
    edge_fn   = EdgeLoss().to(device)
    scaler_G  = GradScaler(enabled=cfg.amp)
    scaler_D  = GradScaler(enabled=cfg.amp)

    out        = Path(cfg.output_dir)
    sample_dir = out / "samples"
    ckpt_dir   = out / "checkpoints"
    writer     = SummaryWriter(out / "tensorboard")

    best_psnr = 0.0; global_step = 0; history: Dict = {}

    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train(); t0 = time.time(); elogs: Dict = {}; nb = 0
        for batch in train_loader:
            logs = train_one_step(batch, G, D, opt_G, opt_D, gan_loss,
                                  perc_loss, edge_fn, scaler_G, scaler_D, device, cfg)
            for k, v in logs.items():
                elogs[k] = elogs.get(k, 0.0) + v
                writer.add_scalar(k, v, global_step)
            global_step += 1; nb += 1
        for k in elogs:
            elogs[k] /= nb
            history.setdefault(k, []).append(elogs[k])
        sch_G.step(); sch_D.step()

        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            metrics = validate(G, val_loader, device, cfg, epoch, sample_dir)
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(f"Epoch [{epoch:03d}/{cfg.epochs}]  "
                  f"G={elogs['loss/G_total']:.4f}  D={elogs['loss/D']:.4f}  "
                  f"PSNR={metrics['PSNR']:.2f}  SSIM={metrics['SSIM']:.4f}  "
                  f"({time.time()-t0:.1f}s)")
            if metrics["PSNR"] > best_psnr:
                best_psnr = metrics["PSNR"]
                save_checkpoint({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
                                 "PSNR": best_psnr, "SSIM": metrics["SSIM"]},
                                str(ckpt_dir / "best_model.pth"))
        else:
            print(f"Epoch [{epoch:03d}/{cfg.epochs}]  "
                  f"G={elogs['loss/G_total']:.4f}  D={elogs['loss/D']:.4f}  "
                  f"({time.time()-t0:.1f}s)")
        if epoch % cfg.save_every == 0:
            save_checkpoint({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict()},
                            str(ckpt_dir / f"epoch_{epoch:04d}.pth"))

    writer.close()
    print(f"\n[done] best PSNR = {best_psnr:.2f} dB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    default="./data")
    p.add_argument("--image_size",   type=int,   default=256)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--ngf",          type=int,   default=64)
    p.add_argument("--ndf",          type=int,   default=64)
    p.add_argument("--n_residual",   type=int,   default=9)
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--beta1",        type=float, default=0.5)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--cpu",          action="store_true")
    p.add_argument("--lambda_gan",   type=float, default=1.0)
    p.add_argument("--lambda_pixel", type=float, default=100.0)
    p.add_argument("--lambda_perc",  type=float, default=10.0)
    p.add_argument("--lambda_edge",  type=float, default=5.0)
    p.add_argument("--output_dir",   default="./runs/sa_gan")
    p.add_argument("--val_every",    type=int,   default=5)
    p.add_argument("--save_every",   type=int,   default=20)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
