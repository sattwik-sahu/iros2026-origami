import torch
import torch.nn.functional as F

from origami_iros.modules.action_head.fm import FlowMatchingActionHead

DIM_ACTION = 65
DIM_IN = 32
N_TOKENS = 6
HIDDEN = 32


def make_head(**overrides):
    kwargs = dict(
        dim_action=DIM_ACTION,
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
    assert out.shape == (4, DIM_ACTION)
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
