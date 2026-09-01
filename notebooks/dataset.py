import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from origami_iros.modules.data.dataset import build_train_val_datasets
    from origami_iros.modules.data.collate import vlta_collate_fn
    from dataclasses import dataclass
    import torch
    from pathlib import Path

    return Path, build_train_val_datasets, dataclass, torch, vlta_collate_fn


@app.cell
def _(dataclass):
    @dataclass
    class TrainConfig:
        data_root: str = "/media/storage/Pranjal/shirish/origami/iros2026-origami/data"
        fps: int = 30

        vit_model_name: str = "facebook/dinov2-small"
        image_size: tuple[int, int] = (480, 480)
        vit_dim: int = 384

        tactile_image_size: tuple[int, int] = (480, 1280)
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

        batch_size: int = 3
        num_workers: int = 8
        lr: float = 1e-4
        lr_min: float = 1e-6
        warmup_steps: int = 1000
        total_steps: int = 20000
        log_every: int = 10
        val_every: int = 1000
        val_batches: int = 20
        ckpt_every: int = 1000
        ckpt_dir: str = "checkpoints"
        val_fraction: float = 0.1
        wandb_project: str = "vlta-flow-matching"
        device: str = "cuda"

    cfg = TrainConfig()
    return (cfg,)


@app.cell
def _(Path, build_train_val_datasets, cfg, torch, vlta_collate_fn):
    device = torch.device(cfg.device)
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    delta_timestamps = {"action": [i / cfg.fps for i in range(cfg.chunk_size)]}
    train_ds, val_ds = build_train_val_datasets(
        cfg.data_root, delta_timestamps, val_fraction=cfg.val_fraction, fps=cfg.fps
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=vlta_collate_fn, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=max(1, cfg.num_workers // 2),
        collate_fn=vlta_collate_fn, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )
    return device, train_loader


@app.cell
def _(train_loader):
    batch = next(iter(train_loader))
    return (batch,)


@app.cell
def _(train_loader):
    batch2 = next(iter(train_loader))
    return


@app.cell
def _(batch):
    obs, action_gt, action_is_pad = batch
    return action_gt, action_is_pad, obs


@app.cell
def _(obs):
    obs
    return


@app.cell
def _(action_gt):
    action_gt.shape
    return


@app.cell
def _(action_gt, cfg):
    from einops import rearrange


    action_gt_reshaped = rearrange(action_gt, "b (n_chunks dim_action) -> b n_chunks dim_action", n_chunks=cfg.chunk_size, dim_action=cfg.action_dim)
    action_gt_reshaped.shape
    return


@app.cell
def _(action_is_pad):
    action_is_pad.shape
    return


@app.cell
def _(action_gt):
    action_gt.shape[-1] // 65
    return


@app.cell
def _(cfg, torch):
    # import torch
    import torch.nn.functional as F

    from origami_iros.modules.action_head.fm import FlowMatchingActionHead

    DIM_ACTION = 65
    DIM_IN = 32
    N_TOKENS = 6
    HIDDEN = 32


    def make_head(**overrides):
        kwargs = dict(
            action_dim=DIM_ACTION,
            dim_in=DIM_IN,
            hidden_dim=HIDDEN,
            num_layers=1,
            num_heads=2,
            num_inference_steps=5,
        )
        kwargs.update(overrides)
        return FlowMatchingActionHead(**kwargs)


    def test_forward_shape():
        head = make_head()
        x = torch.randn(4, N_TOKENS, DIM_IN)
        out = head(x)
        print(out.shape)
        assert out.shape == (4, cfg.chunk_size * DIM_ACTION)
        assert torch.isfinite(out).all()


    def test_compute_loss_shape_and_nonnegative():
        head = make_head()
        x = torch.randn(4, N_TOKENS, DIM_IN)
        target = torch.randn(4, DIM_ACTION)
        loss = head.compute_loss(x, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0


    def test_compute_loss_matches_hand_derived_ot_flow_matching_formula():
        # Re-derive the conditional OT flow matching loss by hand: x_t = (1-t) x0 + t x1,
        # v_target = x1 - x0. If compute_loss matches this exactly, the implementation is
        # really the flow matching objective and not just "some MSE".
        head = make_head()
        head.eval()  # kill dropout randomness so both branches consume RNG identically
        x = torch.randn(3, N_TOKENS, DIM_IN)
        target = torch.randn(3, DIM_ACTION)

        torch.manual_seed(42)
        loss = head.compute_loss(x, target)

        torch.manual_seed(42)
        t = torch.rand(3)
        x0 = torch.randn_like(target)
        t_expanded = t.unsqueeze(-1)
        x_t = (1.0 - t_expanded) * x0 + t_expanded * target
        v_target = target - x0
        v_pred = head.predict_velocity(x_t, t, x)
        expected_loss = F.mse_loss(v_pred, v_target)

        assert torch.allclose(loss, expected_loss, atol=1e-6)


    def test_gradients_reach_every_parameter():
        head = make_head()
        x = torch.randn(4, N_TOKENS, DIM_IN)
        target = torch.randn(4, DIM_ACTION)
        head.compute_loss(x, target).backward()
        missing = [name for name, p in head.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"


    def test_batch_elements_are_independent():
        head = make_head()
        head.eval()
        x = torch.randn(2, N_TOKENS, DIM_IN)
        t = torch.rand(2)
        a = torch.randn(2, DIM_ACTION)

        v_batched = head.predict_velocity(a, t, x)
        v_single_0 = head.predict_velocity(a[:1], t[:1], x[:1])
        v_single_1 = head.predict_velocity(a[1:], t[1:], x[1:])

        assert torch.allclose(v_batched[0], v_single_0[0], atol=1e-5)
        assert torch.allclose(v_batched[1], v_single_1[0], atol=1e-5)


    def test_forward_euler_step_matches_predict_velocity_at_t0():
        # forward() integrates t=0 (noise) -> t=1 (data), one Euler step of size dt=1
        # when num_inference_steps=1: act_1 = act_0 + v_theta(act_0, t=0) * 1.
        head = make_head(num_inference_steps=1)
        x = torch.randn(2, N_TOKENS, DIM_IN)

        torch.manual_seed(7)
        act0 = torch.randn(2, DIM_ACTION)
        t0 = torch.zeros(2)
        v0 = head.predict_velocity(act0, t0, x)
        expected = act0 + v0 * 1.0

        torch.manual_seed(7)
        out = head(x)

        assert torch.allclose(out, expected, atol=1e-5)


    def test_overfits_single_example_and_sampling_converges_to_target():
        # The actual flow-matching check: train on one (obs, target) pair with the
        # conditional OT objective, then confirm samples produced by integrating the
        # learned vector field from noise (forward()) land close to that target --
        # not just that the reported loss number goes down.
        torch.manual_seed(0)
        head = make_head(num_inference_steps=20)
        head.eval()  # disable dropout for a clean, reproducible overfit

        x = torch.randn(1, N_TOKENS, DIM_IN)
        target = torch.randn(1, DIM_ACTION) * 2.0

        opt = torch.optim.Adam(head.parameters(), lr=2e-3)
        losses = []
        for _ in range(400):
            opt.zero_grad()
            loss = head.compute_loss(x, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < 0.05 * losses[0], f"loss barely moved: {losses[0]:.4f} -> {losses[-1]:.4f}"

        with torch.no_grad():
            samples = torch.cat([head(x) for _ in range(16)], dim=0)

        mean_sample = samples.mean(dim=0)
        err = (mean_sample - target[0]).norm().item()
        target_norm = target[0].norm().item()
        print(f"target={target[0].tolist()}")
        print(f"mean of 16 samples={mean_sample.tolist()}")
        print(f"||mean_sample - target|| = {err:.4f} (||target|| = {target_norm:.4f})")

        assert err < 0.35 * target_norm, "sampled actions did not converge toward the trained target"


    def test_more_inference_steps_reduces_discretization_error_on_average():
        # Softer diagnostic, not a hard correctness requirement: with a curved (trained)
        # vector field, a finer Euler integration should track the target at least as
        # well as a single giant step, on average across several noise draws.
        torch.manual_seed(1)
        head = make_head(num_inference_steps=1)
        head.eval()
        x = torch.randn(1, N_TOKENS, DIM_IN)
        target = torch.randn(1, DIM_ACTION) * 2.0

        opt = torch.optim.Adam(head.parameters(), lr=2e-3)
        for _ in range(200):
            opt.zero_grad()
            loss = head.compute_loss(x, target)
            loss.backward()
            opt.step()

        def average_error(num_steps, seeds):
            head.num_inference_steps = num_steps
            errs = []
            with torch.no_grad():
                for seed in seeds:
                    torch.manual_seed(seed)
                    out = head(x)
                    errs.append((out - target).norm().item())
            return sum(errs) / len(errs)

        seeds = list(range(8))
        coarse = average_error(1, seeds)
        fine = average_error(50, seeds)
        print(f"avg err, 1 step={coarse:.4f}, 50 steps={fine:.4f}")

        assert fine < coarse * 1.5, "finer integration should not be dramatically worse than one giant step"


    return (test_forward_shape,)


@app.cell
def _(test_forward_shape):
    test_forward_shape()
    return


@app.cell
def _():
    from origami_iros.modules.encoders import VLTA_Encoder, PretrainedHF_ViT_Encoder, TinyViT_TactileImageEncoder, CameraImageEncoder, TactileImageEncoder
    from origami_iros.modules.encoders.proprio import SingleTokenStateEncoder

    return (
        CameraImageEncoder,
        PretrainedHF_ViT_Encoder,
        SingleTokenStateEncoder,
        TactileImageEncoder,
        TinyViT_TactileImageEncoder,
        VLTA_Encoder,
    )


@app.cell
def _(
    CameraImageEncoder,
    PretrainedHF_ViT_Encoder,
    SingleTokenStateEncoder,
    TactileImageEncoder,
    TinyViT_TactileImageEncoder,
    VLTA_Encoder,
    torch,
):
    encoder = VLTA_Encoder(
        camera_image_encoder=CameraImageEncoder(
            encoder=PretrainedHF_ViT_Encoder(image_size=224, model_name="facebook/dinov2-small", inference_only=True),
        ),
        tactile_image_encoder=TactileImageEncoder(
            deform_encoder=TinyViT_TactileImageEncoder(image_size=(240, 240), patch_size=16, n_hands=2, n_fingers=5)
        ),
        state_encoder=SingleTokenStateEncoder(
            torque_encoder=torch.nn.Linear(65, 128),
            joint_state_encoder=torch.nn.Linear(65, 128),
            tactile_encoder=torch.nn.Linear(60, 128),
            dim=128
        )
    )

    encoder
    return (encoder,)


@app.cell
def _(encoder, obs):
    encoder._camera_image_encoder(obs.image).shape
    return


@app.cell
def _(encoder, obs):
    encoder._tactile_image_encoder(obs.image).shape
    return


@app.cell
def _(encoder, obs):
    encoder._state_encoder(obs.state).shape
    return


@app.cell
def _(encoder, obs):
    encoder(obs)
    return


@app.cell
def _(cfg, device):
    from origami_iros.modules.policy.vlta_policy import VLTAPolicy
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

    return (model,)


@app.cell
def _(model):
    model
    return


@app.cell
def _(model, obs):
    model.sample_actions(obs.cuda()).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
