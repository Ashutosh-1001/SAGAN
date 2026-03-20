from pathlib import Path
import torch
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def denorm(t):
    return ((t + 1) / 2).clamp(0, 1)


def save_comparison_grid(cloudy, fake, clear, path, n_imgs=4):
    n = min(n_imgs, cloudy.size(0))
    rows = []
    for i in range(n):
        rows.append(torch.stack([
            denorm(cloudy[i].cpu()),
            denorm(fake[i].cpu()),
            denorm(clear[i].cpu()),
        ]))
    grid = vutils.make_grid(torch.cat(rows, 0), nrow=3, padding=4, pad_value=1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, path)


def save_attention_map(attn_map, path, overlay=None, alpha=0.5):
    if attn_map.dim() == 4:
        attn_map = attn_map.squeeze(0).squeeze(0)
    attn_np = attn_map.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    if overlay is not None:
        ax.imshow(denorm(overlay.squeeze(0)).permute(1,2,0).cpu().numpy())
        ax.imshow(attn_np, cmap="hot", alpha=alpha, vmin=0, vmax=1)
    else:
        ax.imshow(attn_np, cmap="viridis", vmin=0, vmax=1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_training_curves(history, save_dir, epoch):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loss_keys = [k for k in history if k.startswith("loss/")]
    if loss_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in loss_keys:
            ax.plot(history[k], label=k.replace("loss/", ""))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"losses_epoch{epoch:04d}.png", dpi=120)
        plt.close(fig)
