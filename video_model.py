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
# values from self.frame_model(frames). VideoViT's learned video classifier
# uses temporal transformer outputs only; frame logits are still returned for
# frame/video-mean metrics.


# ---------------------------------------------------------------------------
# Real-video memory bank
# ---------------------------------------------------------------------------

class RealVideoMemoryBank:
    """
    Per-head kNN memory bank of frozen real-video CLS prototypes.

    Stores one mean-pooled CLS prototype per real training video and tapped
    frame-model head. At query time retrieves a weighted real-video prototype
    that is mixed into the current CLS sequence by a learned gate.

    Because these prototypes come from the frozen frame backbone, they are
    stable across Stage 2 training and the bank can be built once.

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
        temperature: float = 0.1,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k         = k
        self.temperature = temperature
        self._keys: List[np.ndarray] = [
            np.zeros((0, embed_dim), dtype=np.float32) for _ in range(num_heads)
        ]
        self._values: List[np.ndarray] = [
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

    def _mean_pool_valid(self, feats: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        if feats.dim() == 2:
            return feats.float()

        if key_padding_mask is None:
            return feats.float().mean(dim=1)

        valid = (~key_padding_mask).to(feats.device).float().unsqueeze(-1)
        counts = valid.sum(dim=1).clamp(min=1)
        return (feats.float() * valid).sum(dim=1) / counts

    def add(self, video_feats_list: List[Tensor], key_padding_mask: Optional[Tensor] = None):
        """
        Add a batch of frozen real-video CLS prototypes to the bank.

        video_feats_list : list of num_heads tensors, each (B, T, embed_dim) or (B, embed_dim)
        """
        for h, feats in enumerate(video_feats_list):
            values = self._mean_pool_valid(feats, key_padding_mask).detach().cpu()
            keys = F.normalize(values.float(), dim=-1).numpy().astype(np.float32)
            values_np = values.float().numpy().astype(np.float32)
            self._keys[h] = np.concatenate([self._keys[h], keys], axis=0)
            self._values[h] = np.concatenate([self._values[h], values_np], axis=0)

    def build(self):
        """Finalise the bank; build FAISS indices if available."""
        n = self._keys[0].shape[0]
        print(f"  [RealVideoMemoryBank] Built with {n} real-video CLS prototypes per head.")

        if self._use_faiss:
            import faiss
            self._faiss_indices = []
            for h in range(self.num_heads):
                index = faiss.IndexFlatIP(self.embed_dim)
                index.add(self._keys[h])
                self._faiss_indices.append(index)

    def __len__(self):
        return self._keys[0].shape[0]

    # ── Querying ────────────────────────────────────────────────────────────

    def query(
        self,
        video_feats_list: List[Tensor],
        key_padding_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Query the bank for each video in the batch.

        video_feats_list : list of num_heads × (B, embed_dim)

        Returns
        -------
        retrieved : list of num_heads tensors, each (B, embed_dim)
        """
        if len(self) == 0:
            raise RuntimeError("RealVideoMemoryBank is empty; call add() and build() first.")

        device = video_feats_list[0].device
        dtype  = video_feats_list[0].dtype
        retrieved = []
        k_eff  = min(self.k, len(self))

        for h, feats in enumerate(video_feats_list):
            pooled = self._mean_pool_valid(feats, key_padding_mask)
            normed = F.normalize(pooled.float(), dim=-1).detach().cpu()

            if self._use_faiss:
                sims, idx = self._faiss_indices[h].search(
                    normed.numpy().astype(np.float32), k_eff
                )
                sims_t = torch.from_numpy(sims)
                values = torch.from_numpy(self._values[h][idx])
            else:
                bank     = torch.from_numpy(self._keys[h])
                sim      = normed @ bank.T
                topk     = sim.topk(k_eff, dim=1)
                sims_t   = topk.values
                values   = torch.from_numpy(self._values[h])[topk.indices]

            weights = torch.softmax(sims_t / self.temperature, dim=1).unsqueeze(-1)
            memory_value = (weights * values).sum(dim=1)
            retrieved.append(memory_value.to(device=device, dtype=dtype))

        return retrieved


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

    def first_layer(
        self,
        frame_cls: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        B, T, _ = frame_cls.shape
        if T > self.num_frames:
            raise ValueError(f"Expected <= {self.num_frames} frames, got {T}")

        x           = frame_cls + self.pos_embed[:, :T, :]
        video_token = self.video_token.expand(B, -1, -1)
        x           = torch.cat([video_token, x], dim=1)

        if key_padding_mask is not None:
            video_mask       = torch.zeros(B, 1, dtype=torch.bool, device=key_padding_mask.device)
            key_padding_mask = torch.cat([video_mask, key_padding_mask], dim=1)

        x = self.encoder.layers[0](x, src_key_padding_mask=key_padding_mask)
        return x, key_padding_mask

    def finish_from_first(
        self,
        x: Tensor,
        full_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.encoder.layers[1:]:
            x = layer(x, src_key_padding_mask=full_padding_mask)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, frame_cls: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x, full_padding_mask = self.first_layer(frame_cls, key_padding_mask)
        return self.finish_from_first(x, full_padding_mask)

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

    VideoViT optionally gates frozen real-video memory prototypes into the
    per-frame CLS sequences, then adds one TemporalTransformer per tapped layer
    (4 total) and fuses everything in a linear classifier.

    Fusion inputs to fusion_classifier
    ------------------------------------
    temporal_vec      : concat of NUM_TEMPORAL_HEADS temporal outputs
                        (4 × 1024 = 4096)
    memory_bank       : optional frozen real-video prototypes gated into CLS tokens
                        before the temporal transformers

    Fusion input dim = 4096, with or without memory.

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
        memory_gate_init: float = 0.12,
    ):
        super().__init__()
        self.num_frames      = num_frames
        self.use_memory_bank = use_memory_bank
        self.memory_bank: Optional[RealVideoMemoryBank] = None
        memory_gate_init = min(max(float(memory_gate_init), 1e-4), 1.0 - 1e-4)
        memory_gate_logit = math.log(memory_gate_init / (1.0 - memory_gate_init))
        self.memory_gate = nn.Parameter(
            torch.full((self.NUM_TEMPORAL_HEADS, 1, 1), memory_gate_logit)
        ) if use_memory_bank else None

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

        self.fusion_classifier = nn.Linear(
            self.NUM_TEMPORAL_HEADS * self.EMBED_DIM, 2
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

    def memory_features_for_bank(
        self,
        video: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> tuple[List[Tensor], Optional[Tensor]]:
        """
        Return per-head frame-token sequences after temporal layer 1.

        Used to rebuild the real-video memory bank at the start of each epoch.
        """
        B, T, C, H, W = video.shape
        if T > self.num_frames:
            raise ValueError(f"Expected <= {self.num_frames} frames, got {T}")

        frames = video.reshape(B * T, C, H, W)
        _, _, cls_list = self.frame_model(frames)

        if lengths is None:
            key_padding_mask = None
        else:
            time_idx = torch.arange(T, device=video.device).unsqueeze(0)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)

        cls_sequences = [
            cls_tokens.reshape(B, T, self.EMBED_DIM) for cls_tokens in cls_list
        ]

        memory_sequences = []
        for temporal_tfm, frame_cls in zip(self.temporal_transformers, cls_sequences):
            x, _ = temporal_tfm.first_layer(frame_cls, key_padding_mask)
            memory_sequences.append(x[:, 1:, :])

        return memory_sequences, key_padding_mask

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        video: Tensor,
        lengths: Optional[Tensor] = None,
        return_memory_ablation: bool = False,
    ):
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
        else:
            time_idx         = torch.arange(T, device=video.device).unsqueeze(0)  # (1, T)
            key_padding_mask = time_idx >= lengths.to(video.device).unsqueeze(1)  # (B, T)

        # ── Temporal transformers ────────────────────────────────────────────
        # cls_list[i] : (B*T, EMBED_DIM) — reshape to (B, T, EMBED_DIM)
        cls_sequences = [
            cls_tokens.reshape(B, T, self.EMBED_DIM) for cls_tokens in cls_list
        ]

        first_outputs = []
        full_masks = []
        memory_query_sequences = []
        for temporal_tfm, frame_cls in zip(self.temporal_transformers, cls_sequences):
            if self.training:
                frame_cls = temporal_augment(frame_cls, key_padding_mask)
            x, full_padding_mask = temporal_tfm.first_layer(frame_cls, key_padding_mask)
            first_outputs.append(x)
            full_masks.append(full_padding_mask)
            memory_query_sequences.append(x[:, 1:, :])

        if self.use_memory_bank:
            assert self.memory_bank is not None, (
                "use_memory_bank=True but no bank attached. "
                "Call attach_memory_bank() first."
            )
            memory_refs = self.memory_bank.query(memory_query_sequences, key_padding_mask)
        else:
            memory_refs = None

        video_feats_list = []
        no_memory_feats_list = [] if return_memory_ablation and memory_refs is not None else None
        for h, (temporal_tfm, x, full_padding_mask) in enumerate(
            zip(self.temporal_transformers, first_outputs, full_masks)
        ):
            if no_memory_feats_list is not None:
                if self.training:
                    with torch.no_grad():
                        no_memory_feats_list.append(
                            temporal_tfm.finish_from_first(x, full_padding_mask)
                        )
                else:
                    no_memory_feats_list.append(
                        temporal_tfm.finish_from_first(x, full_padding_mask)
                    )
            if memory_refs is not None:
                gate = torch.sigmoid(self.memory_gate[h]).to(dtype=x.dtype)
                memory_ref = memory_refs[h].unsqueeze(1)
                frame_tokens = (1 - gate) * x[:, 1:, :] + gate * memory_ref
                x = torch.cat([x[:, :1, :], frame_tokens], dim=1)
            video_feats_list.append(temporal_tfm.finish_from_first(x, full_padding_mask))

        temporal_vec = torch.cat(video_feats_list, dim=1)
        video_logits = self.fusion_classifier(temporal_vec)

        if return_memory_ablation:
            if no_memory_feats_list is None:
                no_memory_logits = video_logits
            else:
                no_memory_vec = torch.cat(no_memory_feats_list, dim=1)
                if self.training:
                    with torch.no_grad():
                        no_memory_logits = self.fusion_classifier(no_memory_vec)
                else:
                    no_memory_logits = self.fusion_classifier(no_memory_vec)
            return (
                video_logits,
                frame_logits_list,
                frame_feats_list,
                video_feats_list,
                no_memory_logits,
            )

        return video_logits, frame_logits_list, frame_feats_list, video_feats_list

        if self.use_memory_bank:
            assert self.memory_bank is not None, (
                "use_memory_bank=True but no bank attached. "
                "Call attach_memory_bank() first."
            )
            memory_refs = self.memory_bank.query(cls_sequences, key_padding_mask)
        else:
            memory_refs = None

        video_feats_list = []
        for h, (temporal_tfm, frame_cls) in enumerate(zip(self.temporal_transformers, cls_sequences)):
            if memory_refs is not None:
                gate = torch.sigmoid(self.memory_gate[h]).to(dtype=frame_cls.dtype)
                memory_ref = memory_refs[h].unsqueeze(1)
                frame_cls = (1 - gate) * frame_cls + gate * memory_ref
            if self.training:
                frame_cls = temporal_augment(frame_cls, key_padding_mask)
            video_feats_list.append(temporal_tfm(frame_cls, key_padding_mask))    # (B, D)

        temporal_vec = torch.cat(video_feats_list, dim=1)                         # (B, 4*1024)

        # ── Fusion classifier input ──────────────────────────────
        fused = temporal_vec                                                       # (B, 4096)

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
