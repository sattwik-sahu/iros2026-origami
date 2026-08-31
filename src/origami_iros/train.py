from dataclasses import dataclass
from pathlib import Path
import time

import torch
import wandb
from torch.utils.data import DataLoader

from origami_iros.modules.policy.vlta_policy import VLTAPolicy
from origami_iros.modules.data.dataset import build_train_val_datasets
from origami_iros.modules.data.collate import vlta_collate_fn


@dataclass
class TrainConfig:
    data_root: str = "/media/storage/Pranjal/shirish/origami/iros2026-origami/data"
    fps: int = 30

    vit_model_name: str = "facebook/dinov2-small"
    image_size: tuple[int, int] = (480, 480)
    vit_dim: int = 384

    tactile_image_size: tuple[int, int] = (480, 1200)
    tactile_patch_size: int = 16
    tactile_dim: int = 192
    n_hands: int = 2
    n_fingers: int = 5

    torque_dim: int = 65
    joint_state_dim: int = 65
    proprio_tactile_dim: int = 60

    hidden_dim: int = 512
    chunk_size: int = 13
    action_dim: int = 65
    action_hidden_dim: int = 512
    action_num_layers: int = 6
    action_num_heads: int = 8
    num_inference_steps: int = 10
    freeze_vit: bool = True

    batch_size: int = 16
    num_workers: int = 8
    lr: float = 1e-4
    lr_min: float = 2.5e-6
    warmup_steps: int = 100
    total_steps: int = 200_000
    log_every: int = 50
    val_every: int = 2_000
    val_batches: int = 20
    ckpt_every: int = 5_000
    ckpt_dir: str = "checkpoints"
    val_fraction: float = 0.1
    wandb_project: str = "vlta-flow-matching"
    device: str = "cuda"


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps), 1.0)
    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)))
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine.item()


@torch.no_grad()
def evaluate(model: VLTAPolicy, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    losses = []
    for i, (obs, target_action, action_is_pad) in enumerate(loader):
        if i >= cfg.val_batches:
            break
        obs = obs.to(device)
        target_action = target_action.to(device)
        action_is_pad = action_is_pad.to(device) if action_is_pad is not None else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model.compute_loss(obs, target_action, action_is_pad)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main(cfg: TrainConfig):
    device = torch.device(cfg.device)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    delta_timestamps = {"action": [i / cfg.fps for i in range(cfg.chunk_size)]}
    train_ds, val_ds = build_train_val_datasets(cfg.data_root, delta_timestamps, val_fraction=cfg.val_fraction)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=vlta_collate_fn, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=max(1, cfg.num_workers // 2),
        collate_fn=vlta_collate_fn, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )

    model = VLTAPolicy(
        vit_model_name=cfg.vit_model_name, image_size=cfg.image_size, vit_dim=cfg.vit_dim,
        tactile_image_size=cfg.tactile_image_size, tactile_patch_size=cfg.tactile_patch_size,
        tactile_dim=cfg.tactile_dim, n_hands=cfg.n_hands, n_fingers=cfg.n_fingers,
        torque_dim=cfg.torque_dim, joint_state_dim=cfg.joint_state_dim,
        proprio_tactile_dim=cfg.proprio_tactile_dim, hidden_dim=cfg.hidden_dim,
        chunk_size=cfg.chunk_size, action_dim=cfg.action_dim, action_hidden_dim=cfg.action_hidden_dim,
        action_num_layers=cfg.action_num_layers, action_num_heads=cfg.action_num_heads,
        num_inference_steps=cfg.num_inference_steps, freeze_vit=cfg.freeze_vit,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95))

    wandb.init(project=cfg.wandb_project, config=cfg.__dict__)

    step = 0
    t0 = time.time()
    model.train()

    while step < cfg.total_steps:
        for obs, target_action, action_is_pad in train_loader:
            if step >= cfg.total_steps:
                break

            obs = obs.to(device)
            target_action = target_action.to(device)
            action_is_pad = action_is_pad.to(device) if action_is_pad is not None else None

            lr = cosine_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model.compute_loss(obs, target_action, action_is_pad)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if step % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(f"→ [{elapsed:8.1f}s] step {step:>7d} | train loss {loss.item():.4f} | lr {lr:.2e}")
                wandb.log({"train/loss": loss.item(), "lr": lr, "step": step})

            if step % cfg.val_every == 0 and step > 0:
                val_loss = evaluate(model, val_loader, cfg, device)
                print(f"✓ [{time.time() - t0:8.1f}s] step {step:>7d} | val loss {val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "step": step})

            if step % cfg.ckpt_every == 0 and step > 0:
                ckpt_path = Path(cfg.ckpt_dir) / f"step_{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
                print(f"✓ saved {ckpt_path}")

            step += 1

    torch.save({"model": model.state_dict(), "step": step}, Path(cfg.ckpt_dir) / "final.pt")
    print("✓ training complete")


if __name__ == "__main__":
    main(TrainConfig(data_root="./data"))