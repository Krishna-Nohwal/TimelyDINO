import timm
import torch
from torch import nn
from peft import LoraConfig, get_peft_model


class AttentionPool(nn.Module):
    """
    Multi-head learnable attention pooling over a token sequence, with an
    optional learned positional bias.

    Each head scores tokens via its own learned query slice (no key/value
    projection — raw token features serve as both K and V, split across
    heads), then returns a per-head softmax-weighted sum, concatenated back
    to the full embed_dim. Using multiple heads instead of one lets the pool
    attend to several distinct spatial patterns at once (e.g. a blending
    artifact in one region and a texture inconsistency in another) rather
    than compromising into a single softmax distribution.

    A learned additive positional bias (one scalar per spatial position) is
    added to the attention logits before softmax, letting each layer learn
    spatial priors (e.g. "blending artifacts cluster near image borders")
    that pure content-based scoring can't capture. The bias is sized for
    `num_patches` tokens; if a forward call provides a different N (e.g.
    a different input resolution under dynamic_img_size), the bias is
    interpolated to match.

    Params: query (embed_dim) + pos_bias (num_patches)

    Input : (B, N, C)
    Output: (B, C)
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_patches: int = 256):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        self.num_patches = num_patches
        self.scale       = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        # One learned bias per spatial position, shared across heads.
        # Zero-init so early training matches plain content-based attention.
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, 1, num_patches))

    def _positional_bias(self, n: int, device, dtype) -> torch.Tensor:
        """Returns (1, 1, 1, n) bias, interpolating if n != num_patches."""
        bias = self.pos_bias
        if n != self.num_patches:
            # Treat as a 1D signal and linearly interpolate to length n.
            bias = nn.functional.interpolate(
                bias.reshape(1, 1, self.num_patches), size=n, mode="linear", align_corners=False
            ).reshape(1, 1, 1, n)
        return bias.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, C)
        Returns:
            pooled : (B, C)
        """
        B, N, C = x.shape
        x_heads = x.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, N, d)
        q = self.query.expand(B, -1, -1, -1)                                        # (B, h, 1, d)

        attn = (q @ x_heads.transpose(-2, -1)) * self.scale                         # (B, h, 1, N)
        attn = attn + self._positional_bias(N, x.device, attn.dtype)                # (B, h, 1, N) broadcast
        attn = attn.softmax(dim=-1)

        out = attn @ x_heads                                                        # (B, h, 1, d)
        return out.permute(0, 2, 1, 3).reshape(B, C)                                # (B, C)


class SpatialHead(nn.Module):
    """
    Spatial Classification head.
    Takes CLS token, REG tokens, and patch tokens from one transformer layer
    and produces logits + a 512-dim intermediate feature vector.

    Single-stage fusion, two-step projection:
        f_cls   : (B, C)  ─┐
        f_reg   : (B, C)  ─┼─ cat → (B, 3C) → head → (B, C//2) → classifier → logits : (B, 2)
        f_patch : (B, C)  ─┘

        f_cls   — global spatial summary (CLS token)
        f_reg   — artifact-localized spatial outliers (mean-pooled)
        f_patch — local spatial features (attention-pooled)

    The head projects gradually instead of collapsing 3C → C//2 in a single
    Linear layer (a 6x reduction in one step): 3C → C → C//2, each stage with
    its own ReLU + Dropout, so the dimensionality reduction is spread out
    rather than sudden.

    Params added per head:
        patch_pool.query     : C            =  1,024
        patch_pool.pos_bias  : num_patches  =    256  (learned spatial prior)
        head[0]              : 3C → C
        head[1]              : C  → C//2

    head input         : 3 * embed_dim  =  3072  →  C  →  C//2   (features)
    logits             : 2  (real / fake)
    """
    def __init__(self, embed_dim: int = 1024, num_reg: int = 4, dropout_p: float = 0.4,
                 num_pool_heads: int = 4, num_patches: int = 256):
        super().__init__()
        self.num_reg    = num_reg

        # Multi-head attention pool with learned positional bias — see
        # AttentionPool docstring for rationale.
        self.patch_pool = AttentionPool(embed_dim, num_heads=num_pool_heads, num_patches=num_patches)
        # reg tokens use simple mean pooling (no learned params)

        # Fuse [f_cls | f_reg | f_patch] → (B, C//2) then classify.
        # Two-step projection (3C → C → C//2) instead of a single 3C → C//2
        # jump, so the dimensionality drop is gradual rather than sudden.
        self.head = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
        self.classifier = nn.Linear(embed_dim // 2, 2)

    def forward(self, cls_tok, reg_tok, patch_tok):
        """
        Args:
            cls_tok   : (B, 1,       embed_dim)
            reg_tok   : (B, num_reg, embed_dim)
            patch_tok : (B, H*W,     embed_dim)
        Returns:
            dict with keys:
                f_cls     : (B, C)
                f_reg     : (B, C)    mean-pooled
                f_patch   : (B, C)    attention-pooled
                logits    : (B, 2)
                features  : (B, C//2 = 512)
        """
        f_cls   = cls_tok.squeeze(1)          # (B, C)
        f_reg   = reg_tok.mean(dim=1)          # (B, C)  — 4 tokens → 1 (mean pool)
        f_patch = self.patch_pool(patch_tok)   # (B, C)  — 256 tokens → 1

        # Single-stage fusion of all three signals, gradual projection
        h = self.head(
            torch.cat([f_cls.float(), f_reg.float(), f_patch.float()], dim=1)  # (B, 3C)
        )                                                                        # (B, C//2)

        return {
            "f_cls":    f_cls,
            "f_reg":    f_reg,
            "f_patch":  f_patch,
            "logits":   self.classifier(h),
            "features": h,
        }


class ViT(nn.Module):
    """
    DINOv3 ViT-Large/16 with 4 register tokens, finetuned with LoRA.

    Forward pass taps intermediate outputs from layers [20, 21, 22, 23] (the
    last 4 transformer blocks). Each layer feeds its own SpatialHead →
    4 sets of (logits, 512-dim features).

    Architecture differences vs ViT-Base:
        - embed_dim 1024 (vs 768)
        - 24 transformer blocks (vs 12)   →  tapped layers: [20, 21, 22, 23]
        - patch size 16 (unchanged)       →  patch grid for 256×256: 16×16 = 256 patches
        - SwiGLU FFN + RoPE positional encoding (same as Base variant)
        - Distilled from 7B teacher on LVD-1689M dataset

    Each SpatialHead's patch-token pooling uses a multi-head AttentionPool
    (4 heads, no K/V projection — split raw tokens across heads) plus a
    learned per-position bias, so each layer can attend to several distinct
    spatial patterns (e.g. blending artifacts vs texture inconsistencies)
    and learn spatial priors (e.g. border vs center) instead of relying on
    a single content-only softmax. See AttentionPool docstring for details.

    Shapes per forward call (batch size B, image size 256×256):
        patch grid    : 16×16 = 256 patches
        prefix_tokens : [CLS, REG_1, REG_2, REG_3, REG_4]  → 5 tokens
        spatial_map   : (B, 1024, 16, 16)
        patch_tok     : (B, 256, 1024)
        cls_tok       : (B, 1,   1024)
        reg_tok       : (B, 4,   1024)

    Returns:
        logits_list   : list of 4 × (B, 2)    — one per tapped layer
        features_list : list of 4 × (B, 512)  — one per tapped layer
        cls_list      : list of 4 × (B, 1024) — one per tapped layer
    """
    EMBED_DIM      = 1024   # ViT-Large hidden size
    NUM_REG        = 4
    NUM_LAYERS     = 4      # number of tapped layers → one SpatialHead each
    LAYERS         = [20, 21, 22, 23]   # last 4 of 24 blocks (0-indexed)
    DROP_PATH      = 0.10
    HEAD_DROP      = 0.4
    NUM_PATCHES    = 256    # 16x16 grid for 256x256 input @ patch size 16
    NUM_POOL_HEADS = 4      # heads in each SpatialHead's AttentionPool

    def __init__(self):
        super().__init__()

        # ── Backbone ────────────────────────────────────────────────────
        self.vit = timm.create_model(
            'vit_large_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=self.DROP_PATH,
        )
        self.vit = get_peft_model(self.vit, LoraConfig(
            r=32,
            lora_alpha=64,        # 2× r
            target_modules=["attn.qkv"],  # verify attr name: may be "qkv" depending on timm version
            lora_dropout=0.10,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ))
        self.vit.base_model.model.set_grad_checkpointing(enable=True)

        # ── One SpatialHead per tapped layer ────────────────────────────
        # Each head's AttentionPool gets its own multi-head query + learned
        # positional bias (sized for NUM_PATCHES; AttentionPool interpolates
        # if a forward call sees a different patch count, e.g. under
        # dynamic_img_size with a non-256 input resolution).
        self.spatial_heads = nn.ModuleList([
            SpatialHead(
                self.EMBED_DIM, self.NUM_REG, self.HEAD_DROP,
                num_pool_heads=self.NUM_POOL_HEADS, num_patches=self.NUM_PATCHES,
            )
            for _ in range(self.NUM_LAYERS)
        ])

    def forward(self, x):
        _, intermediates = self.vit.forward_intermediates(
            x,
            indices=self.LAYERS,
            return_prefix_tokens=True,
            norm=True,
        )

        logits_list:   list = []
        features_list: list = []
        cls_list:      list = []

        for i, (spatial_map, prefix_tokens) in enumerate(intermediates):
            B, C, H, W = spatial_map.shape
            patch_tok = spatial_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
            cls_tok   = prefix_tokens[:, :1, :]
            reg_tok   = prefix_tokens[:, 1:1 + self.NUM_REG, :]

            result = self.spatial_heads[i](cls_tok, reg_tok, patch_tok)
            logits_list.append(result["logits"])
            features_list.append(result["features"])
            cls_list.append(result["f_cls"])

        return logits_list, features_list, cls_list