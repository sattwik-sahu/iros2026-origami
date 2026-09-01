"""Tests for concrete image encoders — shapes, forward pass, and device placement."""

import pytest
import torch

from origami_iros.models._typing import ImageObservation, LeftRightImageObservation, TactileImageObservation
from origami_iros.models.base import BaseImageEncoder, BaseTactileImageEncoder
from origami_iros.models.encoders.image import (
    CameraImageEncoder,
    PerFingerSingleTokenTactileEncoder,
    PretrainedHF_ViT_Encoder,
    TactileImageEncoder,
    TinyViT_TactileImageEncoder,
)

BATCH = 2
C, H, W = 3, 224, 224


class _DummyImageEncoder(BaseImageEncoder):
    """Minimal encoder that produces (B, H*W, C) — for testing wrappers only."""

    def __init__(self):
        super().__init__(image_size=(H, W))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)


class _DummyTactileEncoder(BaseTactileImageEncoder):
    """Minimal tactile encoder that returns a flat encoding per sample."""

    def __init__(self, image_size=(H, W), n_hands=2, n_fingers=4, out_dim=32):
        super().__init__(image_size=image_size, n_hands=n_hands, n_fingers=n_fingers)
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(n_hands * n_fingers, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshaped = self._reshape_image(x)           # (B, n_hands, n_fingers, C, h, w)
        pooled = reshaped.mean(dim=(3, 4, 5))       # (B, n_hands, n_fingers)
        flat = pooled.flatten(1)                     # (B, n_hands*n_fingers)
        return self.linear(flat).unsqueeze(1)        # (B, 1, out_dim)


def _make_image_obs(device):
    return ImageObservation(
        head=LeftRightImageObservation(
            left=torch.randn(BATCH, C, H, W, device=device),
            right=torch.randn(BATCH, C, H, W, device=device),
        ),
        wrist=LeftRightImageObservation(
            left=torch.randn(BATCH, C, H, W, device=device),
            right=torch.randn(BATCH, C, H, W, device=device),
        ),
        tactile=TactileImageObservation(
            deform=torch.randn(BATCH, C, H, W, device=device),
            raw=torch.randn(BATCH, C, H, W, device=device),
        ),
    )


# ── CameraImageEncoder ────────────────────────────────────────────────

class TestCameraImageEncoder:
    def test_output_shape(self, device):
        enc = CameraImageEncoder(encoder=_DummyImageEncoder()).to(device)
        obs = _make_image_obs(device)
        out = enc(obs)
        # 4 views, each producing H*W tokens of dim C → (B, 4*H*W, C)
        assert out.shape == (BATCH, 4 * H * W, C)

    def test_output_dtype(self, device):
        enc = CameraImageEncoder(encoder=_DummyImageEncoder()).to(device)
        obs = _make_image_obs(device)
        out = enc(obs)
        assert out.dtype == torch.float32

    def test_device(self, device):
        enc = CameraImageEncoder(encoder=_DummyImageEncoder()).to(device)
        obs = _make_image_obs(device)
        out = enc(obs)
        assert out.device.type == device.type

    def test_batch_size_1(self, device):
        enc = CameraImageEncoder(encoder=_DummyImageEncoder()).to(device)
        obs = ImageObservation(
            head=LeftRightImageObservation(
                left=torch.randn(1, C, H, W, device=device),
                right=torch.randn(1, C, H, W, device=device),
            ),
            wrist=LeftRightImageObservation(
                left=torch.randn(1, C, H, W, device=device),
                right=torch.randn(1, C, H, W, device=device),
            ),
            tactile=TactileImageObservation(
                deform=torch.randn(1, C, H, W, device=device),
                raw=torch.randn(1, C, H, W, device=device),
            ),
        )
        out = enc(obs)
        assert out.shape[0] == 1


# ── TactileImageEncoder wrapper ───────────────────────────────────────

class TestTactileImageEncoderWrapper:
    def test_raw_output_shape(self, device):
        inner = _DummyTactileEncoder(
            image_size=(H, W), n_hands=2, n_fingers=4, out_dim=32
        ).to(device)
        enc = TactileImageEncoder(primary_encoder=inner).to(device)
        obs = _make_image_obs(device)
        out = enc(obs)
        assert out.shape == (BATCH, 1, 32)

    def test_uses_tactile_raw(self, device):
        """The primary encoder must consume tactile.raw (the real signal)."""
        class RecordingEncoder(BaseTactileImageEncoder):
            def __init__(self):
                super().__init__(image_size=(H, W), n_hands=2, n_fingers=4)
                self.last_input = None

            def forward(self, x):
                self.last_input = x
                return torch.zeros(1)

        recorder = RecordingEncoder().to(device)
        enc = TactileImageEncoder(primary_encoder=recorder).to(device)
        obs = _make_image_obs(device)
        enc(obs)
        assert recorder.last_input is obs.tactile.raw

    def test_uses_deform_when_secondary_provided(self, device):
        """When a secondary encoder is supplied, its tokens are appended."""
        from origami_iros.models.encoders.image import TactileImageEncoder as TE

        class Dummy(BaseTactileImageEncoder):
            def __init__(self, out_dim=8):
                super().__init__(image_size=(H, W), n_hands=2, n_fingers=4)
                self.out_dim = out_dim

            def forward(self, x):
                b = x.shape[0]
                return torch.zeros(b, 1, self.out_dim)

        enc = TE(primary_encoder=Dummy(8), secondary_encoder=Dummy(8), secondary_dropout=0.0)
        enc.eval()
        obs = _make_image_obs(device)
        out = enc(obs)
        # raw (1 token dim 8) concatenated with deform (1 token dim 8) -> (B, 2, 8)
        assert out.shape == (BATCH, 2, 8)


# ── PretrainedHF_ViT_Encoder (DINOv2-small) ───────────────────────────

DINOV2_MODEL = "facebook/dinov2-small"
DINOV2_HIDDEN = 384
DINOV2_PATCH = 14


@pytest.mark.slow
class TestPretrainedHFViTEncoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        return PretrainedHF_ViT_Encoder(
            image_size=H,
            model_name=DINOV2_MODEL,
            inference_only=True,
        )

    def test_construction(self, encoder):
        assert encoder._model_name == DINOV2_MODEL
        assert encoder.image_size == (H, H)

    def test_forward_single_image(self, encoder, device):
        """Test forward pass with a single PIL image."""
        from PIL import Image as PILImage

        model = encoder.to(device)
        img = PILImage.new("RGB", (H, W), color=(100, 120, 140))
        out = model(img)
        assert out.device.type == device.type
        assert out.shape == (1, 256, DINOV2_HIDDEN)

    def test_forward_batch_of_images(self, encoder, device):
        from PIL import Image as PILImage

        model = encoder.to(device)
        imgs = [PILImage.new("RGB", (H, W), color=(i * 30, 100, 200)) for i in range(3)]
        out = model(imgs)
        assert out.shape == (3, 256, DINOV2_HIDDEN)

    def test_cls_token_removed(self, encoder, device):
        """Output must not contain the CLS token (only patch tokens)."""
        from PIL import Image as PILImage

        model = encoder.to(device)
        img = PILImage.new("RGB", (H, W))
        out = model(img)
        # DINOv2 with 224x224 input: 1 CLS + 256 patches = 257 tokens total
        assert out.shape[1] == 256

    def test_inference_mode(self, encoder, device):
        """When inference_only=True, gradients should not be tracked."""
        from PIL import Image as PILImage

        model = encoder.to(device)
        img = PILImage.new("RGB", (H, W))
        out = model(img)
        assert not out.requires_grad

    def test_dtype_is_float32(self, encoder, device):
        from PIL import Image as PILImage

        model = encoder.to(device)
        img = PILImage.new("RGB", (H, W))
        out = model(img)
        assert out.dtype == torch.float32

    def test_model_on_cpu_works(self, encoder):
        """Verify the encoder works correctly on CPU (baseline)."""
        from PIL import Image as PILImage

        model = encoder.to("cpu")
        img = PILImage.new("RGB", (H, W), color=(100, 120, 140))
        out = model(img)
        assert out.shape == (1, 256, DINOV2_HIDDEN)
        assert out.device.type == "cpu"


# ── TinyViT_TactileImageEncoder ───────────────────────────────────────

class TestTinyViT_TactileImageEncoder:
    def test_construction(self):
        enc = TinyViT_TactileImageEncoder(
            image_size=(224, 224), patch_size=4, n_hands=2, n_fingers=4
        )
        assert enc.image_size == (224, 224)
        assert enc._n_hands == 2
        assert enc._n_fingers == 4

    def test_forward_shape(self, device):
        n_hands, n_fingers = 2, 4
        enc = TinyViT_TactileImageEncoder(
            image_size=(H, W), patch_size=4, n_hands=n_hands, n_fingers=n_fingers
        ).to(device)
        x = torch.randn(BATCH, C, H, W, device=device)
        out = enc(x)
        assert out.device.type == device.type
        h_per_finger = H // n_hands   # 112
        w_per_finger = W // n_fingers  # 56
        n_patches = (h_per_finger // 4) * (w_per_finger // 4)
        hidden = enc._model.config.hidden_size
        assert out.shape == (BATCH, n_patches, hidden)

    def test_forward_batch_1(self, device):
        n_hands, n_fingers = 2, 4
        enc = TinyViT_TactileImageEncoder(
            image_size=(H, W), patch_size=4, n_hands=n_hands, n_fingers=n_fingers
        ).to(device)
        x = torch.randn(1, C, H, W, device=device)
        out = enc(x)
        assert out.shape[0] == 1

    def test_model_channels(self, device):
        """Model must have n_hands*n_fingers input channels."""
        n_hands, n_fingers = 2, 4
        enc = TinyViT_TactileImageEncoder(
            image_size=(H, W), patch_size=4, n_hands=n_hands, n_fingers=n_fingers
        ).to(device)
        assert enc._model.config.num_channels == n_hands * n_fingers

    def test_model_image_size(self, device):
        """Model image_size must be square (padded) so the ViT pos-embedding works."""
        n_hands, n_fingers = 2, 4
        enc = TinyViT_TactileImageEncoder(
            image_size=(H, W), patch_size=4, n_hands=n_hands, n_fingers=n_fingers
        ).to(device)
        h_pf, w_pf = H // n_hands, W // n_fingers
        side_patches = max(1, round(((h_pf // 4) * (w_pf // 4)) ** 0.5))
        expected_side = side_patches * 4
        assert enc._model.config.image_size == (expected_side, expected_side)

    def test_various_embodiments(self, device):
        """Encoder must work with different hand/finger counts."""
        for n_h, n_f in [(1, 1), (2, 2), (2, 5), (1, 5)]:
            full_h, full_w = 240 * n_h, 240 * n_f
            enc = TinyViT_TactileImageEncoder(
                image_size=(full_h, full_w), patch_size=4, n_hands=n_h, n_fingers=n_f
            ).to(device)
            x = torch.randn(1, C, full_h, full_w, device=device)
            out = enc(x)
            h_pf = full_h // n_h
            w_pf = full_w // n_f
            n_patches = (h_pf // 4) * (w_pf // 4)
            assert out.shape == (1, n_patches, enc._model.config.hidden_size)


# ── PerFingerSingleTokenTactileEncoder ────────────────────────────────

class TestPerFingerSingleTokenTactileEncoder:
    def test_forward_shape(self, device):
        """Each finger image yields exactly one token of the requested dim."""
        n_hands, n_fingers, token_dim = 2, 5, 64
        full_h, full_w = 480, 1280
        enc = PerFingerSingleTokenTactileEncoder(
            image_size=(full_h, full_w),
            patch_size=16,
            n_hands=n_hands,
            n_fingers=n_fingers,
            token_dim=token_dim,
        ).to(device)
        x = torch.randn(BATCH, C, full_h, full_w, device=device)
        out = enc(x)
        assert out.shape == (BATCH, n_hands * n_fingers, token_dim)
        assert out.device.type == device.type

    def test_fully_connected_gradients(self, device):
        """All parameters must receive gradients after a backward pass."""
        n_hands, n_fingers, token_dim = 2, 2, 32
        full_h, full_w = 240, 240
        enc = PerFingerSingleTokenTactileEncoder(
            image_size=(full_h, full_w),
            patch_size=16,
            n_hands=n_hands,
            n_fingers=n_fingers,
            token_dim=token_dim,
        ).to(device)
        x = torch.randn(2, C, full_h, full_w, device=device)
        out = enc(x)
        out.sum().backward()
        # ViT's unused `mask_token` parameter is idle unless MAE masking is used.
        missing = [
            name
            for name, p in enc.named_parameters()
            if p.grad is None and "mask_token" not in name
        ]
        assert not missing, f"no gradient reached: {missing}"

    def test_batch_1(self, device):
        n_hands, n_fingers = 1, 2
        full_h, full_w = 240, 240
        enc = PerFingerSingleTokenTactileEncoder(
            image_size=(full_h, full_w), patch_size=16, n_hands=n_hands, n_fingers=n_fingers
        ).to(device)
        x = torch.randn(1, C, full_h, full_w, device=device)
        out = enc(x)
        assert out.shape == (1, n_hands * n_fingers, enc._token_dim)

    def test_non_square_finger_padded(self, device):
        """Non-square finger images are padded to a square multiple of patch size."""
        n_hands, n_fingers = 2, 5
        full_h, full_w = 480, 1280  # finger h=240, w=256 (non-square patch counts)
        enc = PerFingerSingleTokenTactileEncoder(
            image_size=(full_h, full_w), patch_size=16, n_hands=n_hands, n_fingers=n_fingers
        ).to(device)
        x = torch.randn(1, C, full_h, full_w, device=device)
        out = enc(x)
        assert out.shape == (1, n_hands * n_fingers, enc._token_dim)
