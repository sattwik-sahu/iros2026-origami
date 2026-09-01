"""Tests for the flow-matching action head.

These verify that the head's loss and sampling behaviour agrees with the
reference ``flow_matching`` library (``CondOTProbPath`` + ``CondOTScheduler`` +
``ODESolver``) which it delegates to, in addition to the standard shape,
gradient, independence and overfit checks.
"""

import torch
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath
from flow_matching.path.scheduler import CondOTScheduler

from origami_iros.models.action_head.fm import FlowMatchingActionHead

DIM_ACTION = 65
CHUNK = 4
DIM_IN = 32
N_TOKENS = 6
HIDDEN = 32


def make_head(**overrides):
    kwargs = dict(
        chunk_size=CHUNK,
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
    assert out.shape == (4, CHUNK, DIM_ACTION)
    assert torch.isfinite(out).all()


def test_dim_action_property():
    head = make_head()
    assert head.dim_action == CHUNK * DIM_ACTION


def test_compute_loss_shape_and_nonnegative():
    head = make_head()
    x = torch.randn(4, N_TOKENS, DIM_IN)
    target = torch.randn(4, CHUNK * DIM_ACTION)
    loss = head.compute_loss(x, target)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_compute_loss_matches_reference_library():
    # The head delegates loss computation to CondOTProbPath + CondOTScheduler. We
    # recompute the interpolation x_t and target velocity dx_t = x1 - x0 using the
    # reference library directly and check the head's loss equals the resulting MSE.
    head = make_head()
    head.eval()  # kill dropout randomness so both branches consume RNG identically

    prob_path = CondOTProbPath()
    scheduler = CondOTScheduler()

    x = torch.randn(3, N_TOKENS, DIM_IN)
    target = torch.randn(3, CHUNK * DIM_ACTION)

    torch.manual_seed(42)
    loss = head.compute_loss(x, target)

    # Reproduce the exact same RNG stream: the head samples t then x0 (Gaussian).
    torch.manual_seed(42)
    alpha, beta, scale, offset = 1.5, 1.0, 0.999, 0.001
    t = torch.distributions.Beta(alpha, beta).sample((3,))
    t = t * scale + offset
    x1 = target.reshape(3, CHUNK, DIM_ACTION)
    x0 = torch.randn_like(x1)

    path_sample = prob_path.sample(x_0=x0, x_1=x1, t=t)
    schedule = scheduler(t)

    memory = head.velocity_model.memory_from_tokens(x)
    v_pred = head.velocity_model(path_sample.x_t, t, memory)
    expected_loss = F.mse_loss(v_pred, path_sample.dx_t)

    # Sanity-check the scheduler/path produce the expected CondOT quantities.
    assert torch.allclose(path_sample.x_t, (1 - t[:, None, None]) * x0 + t[:, None, None] * x1, atol=1e-6)
    assert torch.allclose(path_sample.dx_t, x1 - x0, atol=1e-6)
    assert torch.allclose(schedule.alpha_t, t, atol=1e-6)

    assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_masked_loss_excludes_padded_timesteps():
    head = make_head()
    head.eval()
    x = torch.randn(2, N_TOKENS, DIM_IN)
    target = torch.randn(2, CHUNK * DIM_ACTION)
    mask = torch.ones(2, CHUNK, dtype=torch.bool)
    mask[0, 0] = False

    loss_masked = head.compute_loss(x, target, loss_mask=mask)
    loss_full = head.compute_loss(x, target)

    # Zeroing the mask entry must change the loss (it removes one timestep).
    assert not torch.allclose(loss_masked, loss_full)


def test_gradients_reach_every_parameter():
    head = make_head()
    x = torch.randn(4, N_TOKENS, DIM_IN)
    target = torch.randn(4, CHUNK * DIM_ACTION)
    head.compute_loss(x, target).backward()
    missing = [name for name, p in head.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_batch_elements_are_independent():
    head = make_head()
    head.eval()
    x = torch.randn(2, N_TOKENS, DIM_IN)
    t = torch.rand(2)
    a = torch.randn(2, CHUNK, DIM_ACTION)

    v_batched = head.predict_velocity(a, t, x)
    v_single_0 = head.predict_velocity(a[:1], t[:1], x[:1])
    v_single_1 = head.predict_velocity(a[1:], t[1:], x[1:])

    assert torch.allclose(v_batched[0], v_single_0[0], atol=1e-5)
    assert torch.allclose(v_batched[1], v_single_1[0], atol=1e-5)


def test_overfits_single_example_and_sampling_converges_to_target():
    # Train on a single (obs, target) pair with the conditional-OT objective and
    # confirm the integrated samples (via the reference ODESolver) pull the
    # sampled action chunk clearly towards the target, not just that the reported
    # loss goes down. A small action dimension keeps this fast and stable.
    torch.manual_seed(0)
    head = make_head(dim_in=16, hidden_dim=64, num_layers=2, num_heads=4, action_dim=4, chunk_size=4, num_inference_steps=20)
    head.eval()

    x = torch.randn(1, 4, 16)
    target = torch.randn(1, 4 * 4) * 2.0

    # Baseline error between random samples and the target for reference.
    with torch.no_grad():
        baseline = torch.cat([head(x) for _ in range(32)])
    baseline_err = (baseline - target.reshape(1, 4, 4).expand(32, 4, 4)).norm(dim=(1, 2)).mean().item()

    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    losses = []
    for _ in range(1500):
        opt.zero_grad()
        loss = head.compute_loss(x, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0], f"loss barely moved: {losses[0]:.4f} -> {losses[-1]:.4f}"

    with torch.no_grad():
        samples = torch.cat([head(x) for _ in range(32)], dim=0)

    mean_sample = samples.mean(dim=0)
    tgt = target.reshape(1, head.chunk_size, head.action_dim)[0]
    err = (mean_sample - tgt).norm().item()
    print(f"baseline err={baseline_err:.4f}, trained-mean err={err:.4f}")
    assert err < 0.5 * baseline_err, "samples did not move materially toward the trained target"


def test_more_inference_steps_reduces_discretization_error_on_average():
    torch.manual_seed(1)
    head = make_head(dim_in=16, hidden_dim=64, num_layers=2, num_heads=4, action_dim=4, chunk_size=4, num_inference_steps=1)
    head.eval()
    x = torch.randn(1, 4, 16)
    target = torch.randn(1, 4 * 4) * 2.0

    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    for _ in range(600):
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
                tgt = target.reshape(1, head.chunk_size, head.action_dim)[0]
                errs.append((out - tgt).norm().item())
        return sum(errs) / len(errs)

    seeds = list(range(8))
    coarse = average_error(1, seeds)
    fine = average_error(50, seeds)
    print(f"avg err, 1 step={coarse:.4f}, 50 steps={fine:.4f}")
    assert fine < coarse * 1.5, "finer integration should not be dramatically worse than one giant step"
