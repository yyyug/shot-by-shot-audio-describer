import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from processing.film_grammar import (
    get_effective_shot_scale,
    select_prompt_variant,
    load_dinov2_model,
    ShotScaleClassifier,
    load_shot_scale_classifier,
    extract_middle_frame,
    classify_shot_scale_real,
    predict_threads_real,
    _center_crop_and_resize,
    _softmax_topk,
    _build_neighbor_mask,
)


# ======================================================================
# Original tests
# ======================================================================

class TestGetEffectiveShotScale:
    def test_no_current_shots(self):
        shot_scales = [0, 1, 2, 3, 4]
        result = get_effective_shot_scale(shot_scales, [])
        assert result == 2.0

    def test_with_current_shots(self):
        shot_scales = [0, 1, 2, 3, 4]
        current_shots = [1, 3]
        result = get_effective_shot_scale(shot_scales, current_shots)
        assert result == 2.0

    def test_empty_scales(self):
        result = get_effective_shot_scale([], [0])
        assert result == 2.0

    def test_returns_float(self):
        result = get_effective_shot_scale([1, 2, 3], [0, 1])
        assert isinstance(result, float)


class TestSelectPromptVariant:
    def test_extreme_close_up(self):
        assert select_prompt_variant(0.5) == 1

    def test_close_up(self):
        assert select_prompt_variant(1.2) == 4

    def test_medium(self):
        assert select_prompt_variant(2.0) == 3

    def test_wide(self):
        assert select_prompt_variant(3.2) == 4

    def test_extreme_wide(self):
        assert select_prompt_variant(4.0) == 2

    def test_boundary_low(self):
        assert select_prompt_variant(1.0) == 4

    def test_boundary_medium(self):
        assert select_prompt_variant(1.5) == 3

    def test_boundary_high(self):
        assert select_prompt_variant(3.5) == 2


# ======================================================================
# Helper: create a mock DINOv2 backbone
# ======================================================================

class _FakeBackbone(nn.Module):
    """A lightweight module whose parameters() yields a real CPU tensor."""

    def __init__(self, return_cls_dim: int = 768, spatial_patches: int = 256):
        super().__init__()
        self._dummy = nn.Linear(1, 1)  # gives us a real parameter
        self._cls_dim = return_cls_dim
        self._patches = spatial_patches

    def forward(self, images):
        pass

    def get_intermediate_layers(self, images, n_last_blocks=1, return_class_token=False):
        b = images.shape[0]
        spatial = torch.randn(b, self._patches, self._cls_dim)
        cls = torch.randn(b, self._cls_dim)
        return [(spatial, cls)]


def _make_mock_backbone(return_cls_dim: int = 768, spatial_patches: int = 256):
    """Return a mock DINOv2 backbone whose get_intermediate_layers is tractable."""
    return _FakeBackbone(return_cls_dim, spatial_patches)


# ======================================================================
# load_dinov2_model
# ======================================================================

class TestLoadDinov2Model:
    @patch("processing.film_grammar._setup_and_build_model")
    def test_loads_model(self, mock_setup):
        mock_model = MagicMock()
        mock_setup.return_value = (mock_model, torch.float16)

        model, dtype = load_dinov2_model(
            weights_path="/tmp/fake_weights.pth",
            config_path="/tmp/fake_config.yaml",
        )
        assert model is mock_model
        assert dtype == torch.float16
        mock_setup.assert_called_once()


# ======================================================================
# ShotScaleClassifier
# ======================================================================

class TestShotScaleClassifier:
    def _make_classifier(self):
        backbone = _make_mock_backbone(return_cls_dim=768)
        clf = ShotScaleClassifier(backbone, autocast_dtype=torch.float32, feature_dim=768, num_classes=5)
        clf.eval()
        return clf

    def test_output_shape(self):
        clf = self._make_classifier()
        inp = torch.randn(2, 3, 224, 224)
        out = clf(inp)
        assert out.shape == (2, 5)

    def test_linear_params_exist(self):
        clf = self._make_classifier()
        assert hasattr(clf, "linear")
        assert clf.linear.weight.shape == (5, 768)
        assert clf.linear.bias.shape == (5,)


class TestLoadShotScaleClassifier:
    @patch("processing.film_grammar.load_dinov2_model")
    def test_loads_from_checkpoint(self, mock_load):
        mock_backbone = _make_mock_backbone()
        mock_load.return_value = (mock_backbone, torch.float16)

        fake_state = {"model_state_dict": {"linear.weight": torch.randn(5, 768), "linear.bias": torch.randn(5)}}
        with patch("torch.load", return_value=fake_state):
            clf = load_shot_scale_classifier("/tmp/fake_ckpt.pth")

        assert isinstance(clf, ShotScaleClassifier)
        assert not next(clf.parameters()).requires_grad

    @patch("processing.film_grammar.load_dinov2_model")
    def test_loads_with_explicit_backbone(self, mock_load):
        backbone = _make_mock_backbone()
        fake_state = {"linear.weight": torch.randn(5, 768), "linear.bias": torch.randn(5)}
        with patch("torch.load", return_value=fake_state):
            clf = load_shot_scale_classifier("/tmp/fake_ckpt.pth", backbone=backbone)
        mock_load.assert_not_called()
        assert isinstance(clf, ShotScaleClassifier)


# ======================================================================
# _center_crop_and_resize
# ======================================================================

class TestCenterCropAndResize:
    def test_returns_tensor(self):
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        tensor = _center_crop_and_resize(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_landscape_input(self):
        img = Image.fromarray(np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8))
        tensor = _center_crop_and_resize(img)
        assert tensor.shape == (3, 224, 224)

    def test_portrait_input(self):
        img = Image.fromarray(np.random.randint(0, 255, (640, 360, 3), dtype=np.uint8))
        tensor = _center_crop_and_resize(img)
        assert tensor.shape == (3, 224, 224)

    def test_square_input(self):
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        tensor = _center_crop_and_resize(img)
        assert tensor.shape == (3, 224, 224)


# ======================================================================
# classify_shot_scale_real
# ======================================================================

class TestClassifyShotScaleReal:
    def test_returns_int_in_range(self):
        backbone = _make_mock_backbone()
        clf = ShotScaleClassifier(backbone, autocast_dtype=torch.float32, feature_dim=768, num_classes=5)
        clf.eval()
        frame = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        result = classify_shot_scale_real(frame, backbone, clf)
        assert isinstance(result, int)
        assert 0 <= result <= 4


# ======================================================================
# _softmax_topk
# ======================================================================

class TestSoftmaxTopk:
    def test_output_shape(self):
        sim = torch.randn(4, 256, 256)
        result = _softmax_topk(sim, temperature=0.1, topk=1)
        assert result.shape == (4,)

    def test_values_between_0_and_1(self):
        sim = torch.randn(2, 16, 16)
        result = _softmax_topk(sim, temperature=0.1, topk=1)
        assert (result >= 0).all() and (result <= 1).all()


# ======================================================================
# _build_neighbor_mask
# ======================================================================

class TestBuildNeighborMask:
    def test_output_shape(self):
        mask = _build_neighbor_mask(16, 16, 2)
        assert mask.shape == (256, 256)

    def test_symmetry(self):
        mask = _build_neighbor_mask(4, 4, 1)
        assert torch.all(mask == mask.T)

    def test_diagonal_ones(self):
        mask = _build_neighbor_mask(8, 8, 2)
        assert torch.all(mask.diag() == 1.0)

    def test_reduced_mask_small_neighborhood(self):
        mask = _build_neighbor_mask(16, 16, 0)
        assert mask.shape == (256, 256)
        assert torch.all(mask == torch.eye(256))


# ======================================================================
# predict_threads_real
# ======================================================================

class TestPredictThreadsReal:
    def _make_decord_mock(self, fps=30.0, num_frames=9000):
        """Install a fake decord module into sys.modules for the duration of a test."""
        fake_decord = types.ModuleType("decord")
        fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        class FakeVR:
            def __init__(self, uri=None, ctx=None):
                self._num = num_frames
            def __len__(self):
                return self._num
            def get_avg_fps(self):
                return fps
            def get_batch(self, indices):
                batch = np.array([fake_frame for _ in indices])
                result = MagicMock()
                result.asnumpy.return_value = batch
                return result

        fake_decord.VideoReader = FakeVR
        fake_decord.cpu = lambda x: None

        old = sys.modules.get("decord")
        sys.modules["decord"] = fake_decord
        return old

    def _restore_decord(self, old):
        if old is not None:
            sys.modules["decord"] = old
        else:
            sys.modules.pop("decord", None)

    def test_empty_shots(self):
        backbone = _make_mock_backbone()
        result = predict_threads_real([], "/tmp/fake.mp4", backbone)
        assert result == []

    def test_single_shot(self):
        backbone = _make_mock_backbone()
        old = self._make_decord_mock()
        try:
            result = predict_threads_real(
                [{"start_time": 0.0, "end_time": 1.0}],
                "/tmp/fake.mp4",
                backbone,
            )
            assert result == [[0]]
        finally:
            self._restore_decord(old)

    def test_two_shots_returns_clusters(self):
        backbone = _make_mock_backbone(return_cls_dim=768, spatial_patches=256)
        old = self._make_decord_mock()
        try:
            shots = [
                {"start_time": 0.0, "end_time": 1.0},
                {"start_time": 1.0, "end_time": 2.0},
            ]
            result = predict_threads_real(shots, "/tmp/fake.mp4", backbone, threshold=0.3)
            assert isinstance(result, list)
            assert all(isinstance(t, list) for t in result)
            flat = [idx for thread in result for idx in thread]
            assert sorted(flat) == [0, 1]
        finally:
            self._restore_decord(old)

    def test_three_shots_returns_clusters(self):
        backbone = _make_mock_backbone(return_cls_dim=768, spatial_patches=256)
        old = self._make_decord_mock()
        try:
            shots = [
                {"start_time": 0.0, "end_time": 1.0},
                {"start_time": 1.0, "end_time": 2.0},
                {"start_time": 2.0, "end_time": 3.0},
            ]
            result = predict_threads_real(shots, "/tmp/fake.mp4", backbone, threshold=0.3)
            assert isinstance(result, list)
            flat = [idx for thread in result for idx in thread]
            assert sorted(flat) == [0, 1, 2]
        finally:
            self._restore_decord(old)
