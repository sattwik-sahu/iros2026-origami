"""Tests for abstract base encoder classes and their properties."""

import torch

from origami_iros.modules.base import (
    BaseActionModule,
    BaseEncoder,
    BaseImageEncoder,
    BaseProprioceptiveStateEncoder,
    BaseTactileImageEncoder,
)

BATCH = 2
C, H, W = 3, 224, 224


class _DummyImageEncoder(BaseImageEncoder):
    def __init__(self, image_size):
        super().__init__(image_size=image_size)

    def forward(self, x):
        return x


class _DummyTactileEncoder(BaseTactileImageEncoder):
    def __init__(self, image_size, n_hands, n_fingers):
        super().__init__(image_size=image_size, n_hands=n_hands, n_fingers=n_fingers)

    def forward(self, x):
        return x


class TestBaseImageEncoder:
    def test_image_size_tuple(self):
        enc = _DummyImageEncoder(image_size=(128, 256))
        assert enc.image_size == (128, 256)

    def test_image_size_int(self):
        enc = _DummyImageEncoder(image_size=224)
        assert enc.image_size == (224, 224)

    def test_image_size_property_returns_tuple(self):
        enc = _DummyImageEncoder(image_size=64)
        assert isinstance(enc.image_size, tuple)
        assert len(enc.image_size) == 2


class TestBaseTactileImageEncoder:
    def test_reshape_image(self, device):
        enc = _DummyTactileEncoder(image_size=(224, 224), n_hands=2, n_fingers=4)
        x = torch.randn(BATCH, C, 224, 224, device=device)
        out = enc._reshape_image(x)
        assert out.shape == (BATCH, 2, 4, C, 112, 56)

    def test_reshape_image_1_1(self, device):
        enc = _DummyTactileEncoder(image_size=(128, 64), n_hands=1, n_fingers=1)
        x = torch.randn(BATCH, C, 128, 64, device=device)
        out = enc._reshape_image(x)
        assert out.shape == (BATCH, 1, 1, C, 128, 64)

    def test_reshape_image_square(self, device):
        enc = _DummyTactileEncoder(image_size=(300, 300), n_hands=3, n_fingers=3)
        x = torch.randn(BATCH, C, 300, 300, device=device)
        out = enc._reshape_image(x)
        assert out.shape == (BATCH, 3, 3, C, 100, 100)

    def test_stores_n_hands_n_fingers(self):
        enc = _DummyTactileEncoder(image_size=(224, 224), n_hands=2, n_fingers=5)
        assert enc._n_hands == 2
        assert enc._n_fingers == 5

    def test_image_size(self):
        enc = _DummyTactileEncoder(image_size=(224, 224), n_hands=2, n_fingers=5)
        assert enc.image_size == (224, 224)


class TestBaseActionModule:
    def test_dim_action(self):
        class DummyAction(BaseActionModule):
            def forward(self, x):
                return x

        mod = DummyAction(dim_action=7)
        assert mod.dim_action == 7

    def test_various_dims(self):
        class DummyAction(BaseActionModule):
            def forward(self, x):
                return x

        for d in [1, 7, 14, 32]:
            mod = DummyAction(dim_action=d)
            assert mod.dim_action == d


class TestBaseEncoderInheritance:
    def test_image_encoder_is_encoder(self):
        assert issubclass(BaseImageEncoder, BaseEncoder)

    def test_tactile_encoder_is_image_encoder(self):
        assert issubclass(BaseTactileImageEncoder, BaseImageEncoder)

    def test_state_encoder_is_encoder(self):
        assert issubclass(BaseProprioceptiveStateEncoder, BaseEncoder)

    def test_action_module_is_separate(self):
        assert not issubclass(BaseActionModule, BaseEncoder)
