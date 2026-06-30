import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Optional, List

from frame_model import ViT  # 4-tap backbone: layers [20,21,22,23]


# ---------------------------------------------------------------------------
# Constants derived from the Stage 1 backbone
# ---------------------------------------------------------------------------
# ViT (frame_model_4layers) taps 4 layers [20,21,22,23] and returns 4 entries
# in cls_list / features_list / logits_list — there is no fused_list / 5th
# head and no separate "MACHead" concept; the per-layer head is SpatialHead.
# We drive one TemporalTransformer per tapped layer, so
# NUM_TEMPORAL_HEADS == NUM_LAYERS == 4.
#
# ViT.EMBED_DIM  = 1024  (ViT-Large hidden size)
# ViT.NUM_LAYERS = 4     (number of tapped layers → SpatialHeads)
#
# frame_model_4layers.ViT.forward returns a 3-tuple:
#   (logits_list, features_list, cls_list)
# — NOT a 4-tuple with fused_list, so VideoViT.forward below unpacks 3
# values from self.frame_model(frames), and frame_mean_logits is taken
# from the deepest tap, index 3 (NUM_TEMPORAL_HEADS - 1), not index 4.


# ---------------------------------------------------------------------------
# Real-video memory bank
# ---------------------------------------------------------------------------

class RealVideoMemoryBank:
    """
    Per-head kNN memory bank of real video embeddings.

    Stores L2-normalised embeddings from real training videos, one bank per
    temporal transformer head.  At query time returns the mean cosine
    similarity between the query embedding and its k nearest real neighbours
    — one scalar per head.

    Because the backbone is frozen, embeddings are stable across training so
    the bank is built once before epoch 1 and never updated.

    Uses exact inner-product search over L2-normalised vectors (≡ cosine
    similarity).  FAISS is used when available; falls back to pure-PyTorch
    brute-force otherwise.

    Args
    ----
    embed_dim : int  — embedding dimension (1024 for ViT-Large)
    num_heads : int  — number of temporal transformer heads (== NUM_TEMPORAL_HEADS)
    k         : int  — nearest neighbours to retrieve per query
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 4,   # updated default: 4 tapped layers
        k:         int = 32,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k         = k
        self._banks: List[np.ndarray] = [
            np.zeros((0, embed_dim), dtype=np.float32) for _ in range(num_heads)
        ]
        self._use_faiss     = False
        self._faiss_indices = None

        try:
            import faiss
            self._use_faiss = True
            print("  [RealVideoMemoryBank] FAISS available — using exact IP search.")
        except ImportError:
            print("  [RealVideoMemoryBank] FAISS not found — using PyTorch brute-force search.")

    # ── Building ────────────────────────────────────────────────────────────

    def add(self, video_feats_list: List[Tensor]):
        """
        Add a batch of real video embeddings to the bank.

        video_feats_list : list of num_heads × (B, embed_dim) tensors
        """
        for h, feats in enumerate(video_feats_list):
            normed = F.normalize(feats.float(), dim=-1).detach().cpu().numpy()
            self._banks[h] = np.concatenate([self._banks[h], normed], axis=0)

    def build(self):
        """Finalise the bank; build FAISS indices if available."""
        n = self._banks[0].shape[0]
        print(f"  [RealVideoMemoryBank] Built with {n} real video embeddings per head.")

        if self._use_faiss:
            import faiss
            self._faiss_indices = []
            for h in range(self.num_heads):
                index = faiss.IndexFlatIP(self.embed_dim)
                index.add(self._banks[h])
                self._faiss_indices.append(index)

    def __len__(self):
        return self._banks[0].shape[0]

    # ── Querying ────────────────────────────────────────────────────────────

    def query(self, video_feats_list: List[Tensor]) -> Tensor:
        """
        Query the bank for each video in the batch.

        video_feats_list : list of num_heads × (B, embed_dim)

        Returns
        -------
        sim_scores : (B, num_heads)  mean cosine sim to k nearest real neighbours
        """
        B      = video_feats_list[0].size(0)
        device = video_feats_list[0].device
        scores = []
        k_eff  = min(self.k, len(self))

        for h, feats in enumerate(video_feats_list):
            normed = F.normalize(feats.float(), dim=-1).detach().cpu()

            if self._use_faiss:
                sims, _ = self._faiss_indices[h].search(
                    normed.numpy().astype(np.float32), k_eff
                )
                mean_sim = torch.from_numpy(sims).mean(dim=1)
            else:
                bank     = torch.from_numpy(self._banks[h])
                sim      = normed @ bank.T
                topk     = sim.topk(k_eff, dim=1).values
                mean_sim = topk.mean(dim=1)

            scores.append(mean_sim)

        return torch.stack(scores, dim=1).to(device)   # (B, num_heads)


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def temporal_augment(
    frame_cls: Tensor,
    key_padding_mask: Optional[Tensor],
    blank_prob:   float = 0.15,
    repeat_prob:  float = 0.10,
    noise_std:    float = 0.02,
    mixup_prob:   float = 0.10,
    shuffle_prob: float = 0.05,
    reverse_prob: float = 0.05,
    speed_prob:   float = 0.10,
) -> Tensor:
    """
    Temporal augmentation on CLS-token sequences in embedding space,
    applied after the frozen backbone and before the temporal transformers.

    All augmentations operate on valid (non-padded) frames only.

    Args
    ----
    frame_cls        : (B, T, D)
    key_padding_mask : (B, T) bool — True = padding; None = all valid

    Returns
    -------
    frame_cls : (B, T, D) augmented copy
    """
    B, T, D = frame_cls.shape
    out = frame_cls.clone()

    for b in range(B):
        valid = (
            (~key_padding_mask[b]).nonzero(as_tuple=True)[0]
            if key_padding_mask is not None
            else torch.arange(T, device=frame_cls.device)
        )
        n_valid = valid.numel()
        if n_valid < 2:
            continue

        if torch.rand(1).item() < blank_prob:
            idx = valid[torch.randint(n_valid, (1,)).item()]
            out[b, idx] = 0.0

        if torch.rand(1).item() < repeat_prob and n_valid >= 2:
            src_pos = torch.randint(n_valid, (1,)).item()
            src_idx = valid[src_pos]
            offsets = [-1, 1]
            dst_pos = (src_pos + offsets[torch.randint(2, (1,)).item()]) % n_valid
            dst_idx = valid[dst_pos]
            out[b, dst_idx] = out[b, src_idx].clone()

        if noise_std > 0:
            tokens = out[b, valid]
            norms  = tokens.norm(dim=-1, keepdim=True)
            noise  = torch.randn_like(tokens) * noise_std * norms
            out[b, valid] = tokens + noise

        if torch.rand(1).item() < mixup_prob and n_valid >= 2:
            perm   = torch.randperm(n_valid, device=frame_cls.device)
            i1, i2 = valid[perm[0]], valid[perm[1]]
            lam    = torch.rand(1, device=frame_cls.device).item()
            mixed  = lam * out[b, i1] + (1 - lam) * out[b, i2]
            out[b, i1] = mixed

        if torch.rand(1).item() < shuffle_prob:
            perm          = torch.randperm(n_valid, device=frame_cls.device)
            out[b, valid] = out[b, valid[perm]]

        if torch.rand(1).item() < reverse_prob:
            out[b, valid] = out[b, valid.flip(0)]

        if torch.rand(1).item() < speed_prob and n_valid >= 4:
            rand_pos      = torch.randint(0, n_valid, (n_valid,), device=frame_cls.device)
            rand_pos, _   = rand_pos.sort()
            out[b, valid] = out[b, valid[rand_pos]]

    return out


# ---------------------------------------------------------------------------
# Temporal transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    """Temporal encoder; one CLS-like video token prepended to the frame sequence."""

    def __init__(
        self,
        embed_dim:  int = 1024,
        num_frames: int = 32,
        num_layers: int = 2,
        num_heads:  int = 8,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.num_frames  = num_frames
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.video_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed,   std=0.02)
        nn.init.trunc_normal_(self.video_token, std=0.02)

    def forward(self, frame_cls: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = frame_cls.shape
        if T > self.num_frames:
            raise ValueError(f"Expected ≤{self.num_frames} frames, got {T}")

        x           = frame_cls + self.pos_embed[:, :T, :]
        video_token = self.video_token.expand(B, -1, -1)
        x           = torch.cat([video_token, x], dim=1)

        if key_padding_mask is not None:
            video_mask       = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([video_mask, key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0]   # (B, embed_dim) — video token output


# ---------------------------------------------------------------------------
# VideoViT
# ---------------------------------------------------------------------------

class VideoViT(nn.Module):
    """
    Frame + video deepfake detection model built on top of the Stage 1
    frame_model_4layers.ViT backbone.

    Stage 1 ViT (frame_model_4layers) taps 4 transformer layers [20,21,22,23]
    and returns a 3-tuple:
        logits_list   : 4 × (B*T, 2)     — per-frame classification logits
        features_list : 4 × (B*T, 512)   — 512-dim bottleneck features
        cls_list      : 4 × (B*T, 1024)  — f_cls (CLS token, already squeezed)

    There is no fused_list / spatial_fused output and no separate "MACHead"
    in this backbone — the per-layer classification head is SpatialHead.

    VideoViT adds one TemporalTransformer per tapped layer (4 total), then
    fuses everything in a linear classifier.

    Fusion inputs to fusion_classifier
    ------------------------------------
    temporal_vec      : concat of NUM_TEMPORAL_HEADS temporal outputs
                        (4 × 1024 = 4096)
    frame_mean_logits : mean of deepest-layer (index 3) SpatialHead logits
                        over valid frames (2)
    real_sim_scores   : mean cosine sim to k nearest real videos, one per
                        temporal head (4) — ONLY when memory_bank is set

    Without memory bank : fusion input dim = 4096 + 2      = 4098
    With    memory bank : fusion input dim = 4096 + 2 + 4  = 4102

    The memory_bank is NOT a nn.Module and is NOT in state_dict; it must be
    rebuilt and re-attached after loading a checkpoint.
    """

    EMBED_DIM          = ViT.EMBED_DIM   # 1024
    NUM_TEMPORAL_HEADS = ViT.NUM_LAYERS  # 4  (one per tapped layer)

    def __init__(
        self,
        num_frames:       int   = 32,
        temporal_layers:  int   = 2,
        temporal_heads:   int   = 8,
        temporal_dropout: float = 0.1,
        use_memory_bank:  bool  = False,
    ):
        super().__init__()
        self.num_frames      = num_frames
        self.use_memory_bank = use_memory_bank
        self.memory_bank: Optional[RealVideoMemoryBank] = None

        self.frame_model = ViT()

        self.temporal_transformers = nn.ModuleList([
            TemporalTransformer(
                embed_dim  = self.EMBED_DIM,
                num_frames = num_frames,
                num_layers = temporal_layers,
                num_heads  = temporal_heads,
                dropout    = temporal_dropout,
            )
            for _ in range(self.NUM_TEMPORAL_HEADS)
        ])

        knn_dim = self.NUM_TEMPORAL_HEADS if use_memory_bank else 0
        self.fusion_classifier = nn.Linear(
            self.NUM_TEMPORAL_HEADS * self.EMBED_DIM + 2 + knn_dim, 2
        )

    # ── Compatibility alias (old code referenced model.vit) ─────────────────
    @property
    def vit(self):
        return self.frame_model.vit

    # ── Memory bank ─────────────────────────────────────────────────────────

    def attach_memory_bank(self, bank: RealVideoMemoryBank):
        assert self.use_memory_bank, \
            "VideoViT was not constructed with use_memory_bank=True."
        assert bank.num_heads == self.NUM_TEMPORAL_HEADS, (
            f"Bank has {bank.num_heads} heads but model expects {self.NUM_TEMPORAL_HEADS}."
        )
        self.memory_bank = bank

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, video: Tensor, lengths: Optional[Tensor] = None):
        """
        Args
        ----
        video   : (B, T, C, H, W)
        lengths : (B,) int — actual frame count per clip (None → all T)

        Returns
        -------
        video_logits      : (B, 2)
        frame_logits_list : list of 4 × (B*T, 2)
        frame_feats_list  : list of 4 × (B*T, 512)
        video_feats_list  : list of 4 × (B, 1024)
        """
        B, T, C, H, W = video.shape
        if T > self.num_frames:
            raise ValueError(f"Expected ≤{self.num_frames} frames, got {T}")

        frames = video.reshape(B * T, C, H, W)

        # frame_model_4layers.ViT returns (logits_list, features_list, cls_list)
        # — a 3-tuple, no fused_list. cls_list entries are (B*T, EMBED_DIM)
        # — f_cls already squeezed by SpatialHead.
        frame_logits_list, frame_feats_list, cls_list = self.frame_model(frames)

        # ── Padding mask ────────────────────────────────────────────────────
        if lengths is None:
            key_padding_mask = None
            valid_counts     = torch.full((B,), T, dtype=torch.float32, device=video.device)
        else:
            time_idx         = torch.arange(T, device=video.device).unsqueeze(0)  # (1, T)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)  # (B, T)
            valid_counts     = lengths.to(video.device).float()

        # ── frame_mean_logits from deepest SpatialHead (index NUM_TEMPORAL_HEADS - 1 = 3) ──
        # frame_logits_list[3] : (B*T, 2)  — deepest tap, layer 23
        deepest_idx       = self.NUM_TEMPORAL_HEADS - 1
        frame_logits_bt   = frame_logits_list[deepest_idx].reshape(B, T, 2).float()  # (B, T, 2)
        if key_padding_mask is not None:
            valid_mask        = (~key_padding_mask).float().unsqueeze(-1)          # (B, T, 1)
            frame_logits_bt   = frame_logits_bt * valid_mask
        frame_mean_logits = (
            frame_logits_bt.sum(dim=1) /
            valid_counts.unsqueeze(1).clamp(min=1)
        )                                                                           # (B, 2)

        # ── Temporal transformers ────────────────────────────────────────────
        # cls_list[i] : (B*T, EMBED_DIM) — reshape to (B, T, EMBED_DIM)
        video_feats_list = []
        for temporal_tfm, cls_tokens in zip(self.temporal_transformers, cls_list):
            frame_cls = cls_tokens.reshape(B, T, self.EMBED_DIM)                  # (B, T, D)
            if self.training:
                frame_cls = temporal_augment(frame_cls, key_padding_mask)
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))    # (B, D)

        temporal_vec = torch.cat(video_feats_list, dim=1)                         # (B, 4*1024)

        # ── Optional kNN real-video similarity ──────────────────────────────
        if self.use_memory_bank:
            assert self.memory_bank is not None, (
                "use_memory_bank=True but no bank attached. "
                "Call attach_memory_bank() first."
            )
            real_sim = self.memory_bank.query(video_feats_list)                   # (B, 4)
            fused    = torch.cat([temporal_vec, frame_mean_logits, real_sim], dim=1)
        else:
            fused    = torch.cat([temporal_vec, frame_mean_logits], dim=1)        # (B, 4098)

        video_logits = self.fusion_classifier(fused)                              # (B, 2)

        return video_logits, frame_logits_list, frame_feats_list, video_feats_list

    # ── Checkpoint loading ───────────────────────────────────────────────────

    def load_image_weights(self, image_ckpt_path: str, strict: bool = False):
        """
        Load Stage 1 weights into frame_model.

        Handles two checkpoint formats:
          (a) Full VideoViT state_dict  → keys prefixed with 'frame_model.'
          (b) Bare ViT state_dict       → keys NOT prefixed with 'frame_model.'
              (this is the format produced by train_stage1_4layers.py best.pth)
        """
        ckpt  = torch.load(image_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))

        # Separate frame_model keys from temporal / fusion keys.
        frame_state   = {}
        has_fm_prefix = any(k.startswith("frame_model.") for k in state)

        for key, value in state.items():
            if key.startswith("frame_model."):
                frame_state[key[len("frame_model."):]] = value
            elif key.startswith(("temporal_transformers.", "fusion_classifier.")):
                pass  # skip video-level weights from a Stage 2 checkpoint
            else:
                if not has_fm_prefix:
                    # Bare Stage 1 checkpoint — all keys belong to frame_model.
                    frame_state[key] = value

        missing, unexpected = self.frame_model.load_state_dict(frame_state, strict=strict)
        print(f"Loaded image weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
        return missing, unexpected