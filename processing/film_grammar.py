"""
Film grammar utilities - supplements original repo's film_grammar modules.
Includes DINOv2-based shot scale classification and thread structure prediction.
"""
import sys
from typing import List, Dict, Optional, Tuple
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Original utility functions
# ---------------------------------------------------------------------------

def get_effective_shot_scale(shot_scales: List[int], current_shots: List[int]) -> float:
    """
    Calculate effective shot scale for prompt selection.
    
    Args:
        shot_scales: List of all shot scales
        current_shots: Indices of current shots
        
    Returns:
        Average shot scale of current shots
    """
    if not current_shots:
        return np.mean(shot_scales) if shot_scales else 2.0
    
    current_scales = [shot_scales[i] for i in current_shots if i < len(shot_scales)]
    return np.mean(current_scales) if current_scales else 2.0


def select_prompt_variant(effective_scale: float) -> int:
    """
    Select prompt variant based on effective shot scale.
    
    Args:
        effective_scale: Average shot scale
        
    Returns:
        Prompt variant (1-4)
    """
    if effective_scale < 1.0:
        return 1  # facial expressions
    elif effective_scale >= 3.5:
        return 2  # environment
    elif 1.5 <= effective_scale < 3.0:
        return 3  # key objects
    else:
        return 4  # none (3 steps only)


# ---------------------------------------------------------------------------
# DINOv2 model loading
# ---------------------------------------------------------------------------

_DINOV2_CONFIG = str(
    Path(__file__).resolve().parent.parent
    / "film_grammar" / "dinov2" / "dinov2" / "configs" / "eval" / "vitb14_pretrain.yaml"
)
_DEFAULT_DINOV2_WEIGHTS = str(
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
)


def _build_dinov2_args(weights_path: str, config_path: str):
    """Build the minimal args namespace required by setup_and_build_model."""

    class _Args:
        pass

    args = _Args()
    args.pretrained_weights = weights_path
    args.config_file = config_path
    args.output_dir = ""
    defaults = {
        "train_dataset_str": "ImageNet:split=TRAIN",
        "val_dataset_str": "ImageNet:split=VAL",
        "nb_knn": [10],
        "temperature": 0.07,
        "gather_on_cpu": False,
        "batch_size": 256,
        "n_per_class_list": [-1],
        "n_tries": 1,
        "seed": 0,
        "opts": [],
    }
    for attr, val in defaults.items():
        if not hasattr(args, attr):
            setattr(args, attr, val)
    return args


def _setup_and_build_model(args):
    """
    Thin wrapper around the DINOv2 eval setup.
    Exists as a module-level callable so tests can patch it.
    """
    _dinov2_root = str(Path(__file__).resolve().parent.parent / "film_grammar" / "dinov2")
    if _dinov2_root not in sys.path:
        sys.path.insert(0, _dinov2_root)
    from dinov2.eval.setup import setup_and_build_model  # type: ignore
    return setup_and_build_model(args)


def load_dinov2_model(
    weights_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, torch.dtype]:
    """
    Load DINOv2 ViT-B/14 for feature extraction.

    Returns:
        (model, autocast_dtype) – the backbone in eval mode on the current device
        and the dtype to use for mixed-precision inference.
    """
    weights_path = weights_path or _DEFAULT_DINOV2_WEIGHTS
    config_path = config_path or _DINOV2_CONFIG

    args = _build_dinov2_args(weights_path, config_path)
    model, autocast_dtype = _setup_and_build_model(args)
    return model, autocast_dtype


# ---------------------------------------------------------------------------
# Shot scale classifier
# ---------------------------------------------------------------------------

class ShotScaleClassifier(nn.Module):
    """
    DINOv2 ViT-B/14 backbone + linear classifier head for shot scale.

    Scale labels: 0 = extreme close-up, 1 = close-up, 2 = medium,
                  3 = wide, 4 = extreme wide.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        autocast_dtype: torch.dtype = torch.float16,
        feature_dim: int = 768,
        num_classes: int = 5,
    ):
        super().__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
        self.autocast_ctx = partial(
            torch.amp.autocast, 'cuda', enabled=True, dtype=autocast_dtype,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.backbone.get_intermediate_layers(
                    images, n_last_blocks=1, return_class_token=True,
                )
        cls_token = features[0][1].clone()  # clone to detach from inference context
        return self.linear(cls_token)


def load_shot_scale_classifier(
    model_path: str,
    backbone: Optional[torch.nn.Module] = None,
    autocast_dtype: torch.dtype = torch.float16,
    feature_dim: int = 768,
    num_classes: int = 5,
) -> ShotScaleClassifier:
    """
    Load a trained shot scale classifier on top of a DINOv2 backbone.

    Args:
        model_path: Path to the classifier checkpoint (.pth).
        backbone: Pre-loaded DINOv2 backbone. If *None*, one is loaded
                  automatically via :func:`load_dinov2_model`.
        autocast_dtype: Mixed-precision dtype for the backbone.
        feature_dim: Backbone output dimension (768 for ViT-B/14).
        num_classes: Number of shot scale categories.

    Returns:
        The classifier in eval mode.
    """
    if backbone is None:
        backbone, autocast_dtype = load_dinov2_model()

    classifier = ShotScaleClassifier(backbone, autocast_dtype, feature_dim, num_classes)
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        classifier.load_state_dict(checkpoint, strict=False)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_middle_frame(video_path: str, shot: dict) -> Image.Image:
    """
    Extract the middle frame from a shot within a video.

    Args:
        video_path: Path to the video file.
        shot: Shot dict with keys ``start_time`` and ``end_time`` (seconds).

    Returns:
        The middle frame as a PIL Image.
    """
    from decord import VideoReader, cpu

    vr = VideoReader(uri=video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    start_frame = int(shot["start_time"] * fps)
    end_frame = int(shot["end_time"] * fps)
    end_frame = min(end_frame, len(vr) - 1)
    mid_frame = (start_frame + end_frame) // 2
    mid_frame = max(0, min(mid_frame, len(vr) - 1))
    frame = vr.get_batch([mid_frame]).asnumpy()[0]
    return Image.fromarray(frame)


# ---------------------------------------------------------------------------
# Image pre-processing for DINOv2
# ---------------------------------------------------------------------------

def _center_crop_and_resize(
    img: Image.Image,
    target_size: int = 224,
    aspect_ratio: float = 4 / 3,
) -> torch.Tensor:
    """
    Center-crop to *aspect_ratio* then resize to (*target_size* × *target_size*).
    Returns a normalised tensor ready for DINOv2.
    """
    w, h = img.size
    if w / h > aspect_ratio:
        new_h = h
        new_w = int(h * aspect_ratio)
    else:
        new_w = w
        new_h = int(w / aspect_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((target_size, target_size), Image.BICUBIC)
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)


# ---------------------------------------------------------------------------
# Shot scale classification
# ---------------------------------------------------------------------------

def classify_shot_scale_real(
    frame: Image.Image,
    model: torch.nn.Module,
    classifier: torch.nn.Module,
) -> int:
    """
    Classify the shot scale of a single frame.

    Args:
        frame: PIL Image (any resolution).
        model: DINOv2 backbone (used for reference, not directly).
        classifier: :class:`ShotScaleClassifier` wrapping the backbone.

    Returns:
        Predicted scale label (0–4).
    """
    device = next(classifier.parameters()).device
    tensor = _center_crop_and_resize(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(tensor)
    return int(torch.argmax(logits, dim=1).item())


# ---------------------------------------------------------------------------
# Thread structure prediction
# ---------------------------------------------------------------------------

def _softmax_topk(cos_sim: torch.Tensor, temperature: float, topk: int) -> torch.Tensor:
    """Softmax temperature scaling + top-k pooling."""
    cos_sim = (cos_sim / temperature).softmax(-1)
    tk_val, _ = torch.topk(cos_sim, dim=-1, k=topk)
    return tk_val.mean(-1).mean(-1)


def _build_neighbor_mask(
    h: int, w: int, size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a spatial locality mask for patch-wise cosine similarity."""
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size + 1):
                for q in range(2 * size + 1):
                    ni = i - size + p
                    nj = j - size + q
                    if 0 <= ni < h and 0 <= nj < w:
                        mask[i, j, ni, nj] = 1
    return mask.reshape(h * w, h * w).to(device)


def predict_threads_real(
    shots: List[dict],
    video_path: str,
    model: torch.nn.Module,
    threshold: float = 0.3,
    temperature: float = 0.1,
    topk: int = 1,
    size_mask_neighborhood: int = 2,
) -> List[List[int]]:
    """
    Predict thread structure for a sequence of shots using DINOv2 features.

    Each thread is a list of shot indices that share visual continuity.

    Args:
        shots: Sequence of shot dicts with ``start_time`` and ``end_time`` keys.
        video_path: Path to the video file.
        model: DINOv2 backbone (in eval mode).
        threshold: Cosine-similarity threshold for adjacency.
        temperature: Softmax temperature for top-k pooling.
        topk: Number of top elements to pool.
        size_mask_neighborhood: Spatial neighbourhood radius.

    Returns:
        List of threads (each thread is a sorted list of shot indices).
    """
    if len(shots) == 0:
        return []
    if len(shots) == 1:
        return [[0]]

    from decord import VideoReader, cpu

    device = next(model.parameters()).device
    vr = VideoReader(uri=video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())

    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract start & end frames for every shot
    shot_frames: list[Image.Image] = []
    for shot in shots:
        sf = int(shot["start_time"] * fps)
        ef = min(int(shot["end_time"] * fps), len(vr) - 1)
        sf = max(0, sf)
        ef = max(sf, ef)
        frame_s = Image.fromarray(vr.get_batch([sf]).asnumpy()[0])
        frame_e = Image.fromarray(vr.get_batch([ef]).asnumpy()[0])
        shot_frames.extend([frame_s, frame_e])

    # Build tensors and extract DINOv2 features
    tensors = torch.stack([transform(f) for f in shot_frames]).to(device)
    with torch.inference_mode():
        features = model.get_intermediate_layers(
            tensors, n_last_blocks=1, return_class_token=True,
        )
    # features[0] = (spatial, cls) – take CLS + spatial patches
    spatial, cls = features[0]  # spatial: (B, patch_h*patch_w, dim), cls: (B, dim)
    feat_all = torch.cat([cls.unsqueeze(1), spatial], dim=1)  # (B, 1+patches, dim)

    # Separate start/end features
    feat_start = feat_all[0::2]  # (N, P, D)
    feat_end = feat_all[1::2]    # (N, P, D)

    # Build all temporally-adjacent pairs
    pairs = [(i, j) for i in range(len(shots)) for j in range(i + 1, len(shots))]
    feature_pairs = []
    for i, j in pairs:
        feature_pairs.append(torch.stack([feat_end[i], feat_start[j]], dim=0))
    feature_pairs = torch.stack(feature_pairs, dim=0)  # (num_pairs, 2, P, D)
    feature_pairs = F.normalize(feature_pairs, p=2, dim=-1)

    features_left = feature_pairs[:, 0, 1:, :].to(device)   # (num_pairs, P, D)
    features_right = feature_pairs[:, 1, 1:, :].to(device)  # (num_pairs, P, D)

    # Cosine similarity + spatial masking
    cos_sim = torch.bmm(features_left, features_right.transpose(1, 2))

    # Determine patch grid from spatial feature size
    num_patches = features_left.shape[1]
    patch_h = int(np.sqrt(num_patches))
    patch_w = num_patches // patch_h
    neighbor_mask = _build_neighbor_mask(
        patch_h, patch_w, size_mask_neighborhood, device,
    )

    cos_sim = cos_sim * neighbor_mask.unsqueeze(0)

    cos_sim_left = _softmax_topk(cos_sim, temperature, topk)
    cos_sim_right = _softmax_topk(cos_sim.transpose(-1, -2), temperature, topk)
    cos_sim_comb = (cos_sim_left + cos_sim_right) / 2

    # Threshold -> adjacency matrix -> connected components
    import networkx as nx

    adj = np.eye(len(shots), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        if cos_sim_comb[idx].item() > threshold:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    G = nx.from_numpy_array(adj)
    clusters = [sorted(c) for c in nx.connected_components(G)]
    return clusters
