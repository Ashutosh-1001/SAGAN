import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from models import Generator
from utils.metrics import psnr, ssim
from utils.visualize import save_comparison_grid, save_attention_map

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_normalize = T.Normalize([0.5]*3, [0.5]*3)
_denorm = lambda t: ((t.clamp(-1,1)+1)/2)


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return _normalize(TF.to_tensor(img)).unsqueeze(0)


def tensor_to_pil(t):
    arr = (_denorm(t.squeeze(0)).permute(1,2,0).cpu().numpy()*255).astype("uint8")
    return Image.fromarray(arr)


class AttentionHook:
    def __init__(self, module):
        self.maps = []
        self._hook = module.register_forward_hook(
            lambda m, i, o: self.maps.append(o.detach().cpu()))
    def clear(self): self.maps.clear()
    def remove(self): self._hook.remove()


@torch.no_grad()
def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    G = Generator(ngf=cfg.ngf, n_residual=cfg.n_residual).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    G.load_state_dict(ckpt.get("G", ckpt)); G.eval()

    attn_hook = None
    if cfg.save_attention:
        from models.attention import SpatialAttention
        for m in G.modules():
            if isinstance(m, SpatialAttention):
                attn_hook = AttentionHook(m); break

    input_dir = Path(cfg.input)
    output_dir = Path(cfg.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
    clear_dir = input_dir.parent / "clear"
    compute = cfg.compute_metrics and clear_dir.exists()

    total_p = total_s = n = 0.0
    for img_path in paths:
        cloudy_t = load_image(str(img_path), cfg.image_size).to(device)
        if attn_hook: attn_hook.clear()
        fake_t = G(cloudy_t)
        tensor_to_pil(fake_t).save(str(output_dir / img_path.name))

        if attn_hook and attn_hook.maps:
            am = attn_hook.maps[0].mean(dim=1, keepdim=True)
            am = (am - am.min()) / (am.max() - am.min() + 1e-8)
            ap = output_dir / "attention" / img_path.stem
            ap.parent.mkdir(parents=True, exist_ok=True)
            save_attention_map(am, str(ap)+"_heat.png", overlay=cloudy_t)

        if compute:
            cp = clear_dir / img_path.name
            if cp.exists():
                ct = load_image(str(cp), cfg.image_size).to(device)
                total_p += psnr(fake_t, ct)
                total_s += ssim(fake_t, ct)
                n += 1
                if cfg.save_comparison:
                    cmp = output_dir / "comparisons" / img_path.name
                    cmp.parent.mkdir(parents=True, exist_ok=True)
                    save_comparison_grid(cloudy_t, fake_t, ct, str(cmp), n_imgs=1)
        print(f"{img_path.name}")

    if compute and n:
        print(f"\nPSNR={total_p/n:.2f} dB  SSIM={total_s/n:.4f}")
    if attn_hook: attn_hook.remove()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="./results")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--n_residual", type=int, default=9)
    p.add_argument("--save_attention", action="store_true")
    p.add_argument("--save_comparison", action="store_true")
    p.add_argument("--compute_metrics", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
