import argparse
import torch
from tqdm import tqdm
from models import Generator
from data import get_dataloader
from utils import MetricTracker


@torch.no_grad()
def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    G = Generator(ngf=cfg.ngf, n_residual=cfg.n_residual).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    G.load_state_dict(ckpt.get("G", ckpt)); G.eval()

    loader = get_dataloader(cfg.data_root, cfg.split, cfg.image_size, cfg.batch_size)
    tracker = MetricTracker()
    for batch in tqdm(loader, desc=f"Evaluating [{cfg.split}]"):
        cloudy = batch["cloudy"].to(device, non_blocking=True)
        clear  = batch["clear"].to(device, non_blocking=True)
        tracker.update(G(cloudy), clear)

    s = tracker.summary()
    print(f"\n{'='*40}\n  Split : {cfg.split}")
    print(f"  Images: {len(loader.dataset)}")
    print(f"  PSNR  : {s['PSNR']:.2f} dB")
    print(f"  SSIM  : {s['SSIM']:.4f}\n{'='*40}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--split", default="test")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--n_residual", type=int, default=9)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
