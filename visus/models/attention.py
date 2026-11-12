"""
Read CLIP's internals: the CLS->patch attention of the last vision block, and text features.

Two versions exist, selected by `ModelConfig` (see model.py):

* v2 (current): attention is computed from the LayerNorm-ed block input, exactly as CLIP
  does; gradients flow through it so the last vision blocks really fine-tune; the
  prompt is represented by CLIP's text embedding (512-d, frozen).
* v1 (legacy, the released `visus_epoch50.pt`): attention computed on the raw block
  input (no LayerNorm), everything under `no_grad`, and the prompt reduced to one
  attention statistic per text layer. Kept only so old checkpoints reproduce.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F


def _attention_from_block(module, x: torch.Tensor, use_layernorm: bool) -> torch.Tensor:
    """Recompute the softmax attention of a CLIP ResidualAttentionBlock. x: (L, B, D)."""
    if use_layernorm:
        x = module.ln_1(x)
    B = x.shape[1]
    qkv = F.linear(x, module.attn.in_proj_weight, module.attn.in_proj_bias)
    q, k, _ = qkv.chunk(3, dim=-1)
    num_heads = module.attn.num_heads
    head_dim = q.shape[-1] // num_heads
    q = q.reshape(q.shape[0], B, num_heads, head_dim).permute(1, 2, 0, 3)
    k = k.reshape(k.shape[0], B, num_heads, head_dim).permute(1, 2, 0, 3)
    attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    return attn.softmax(dim=-1)                               # (B, H, L, L)


class AttentionExtractor:
    """CLS->patch attention of the last visual block, as a (B, 1, H, W) map in [0, 1]."""

    def __init__(self, model, device: str = "cpu", use_layernorm: bool = True, trainable: bool = True):
        self.model = model
        self.device = device
        self.use_layernorm = use_layernorm
        self.trainable = trainable
        self.attn_maps: list[torch.Tensor] = []
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        last_block = self.model.visual.transformer.resblocks[-1]

        def _hook_fn(module, inputs, output):
            attn = _attention_from_block(module, inputs[0], self.use_layernorm)
            self.attn_maps.append(attn if self.trainable else attn.detach())

        self._hook = last_block.register_forward_hook(_hook_fn)

    def get_attention_map(self, image: torch.Tensor, target_size: tuple[int, int] = (224, 224)) -> torch.Tensor:
        self.attn_maps.clear()
        ctx = contextlib.nullcontext() if self.trainable else torch.no_grad()
        with ctx:
            self.model.encode_image(image.to(self.device))
        if not self.attn_maps:
            raise RuntimeError("No attention maps captured!")

        attn = self.attn_maps[-1].float()                      # (B, H, L, L)
        B = attn.shape[0]
        cls_attn = attn[:, :, 0, 1:].mean(1)                   # (B, num_patches)
        grid = int(np.sqrt(cls_attn.shape[-1]))
        if grid * grid != cls_attn.shape[-1]:
            raise ValueError(f"Patch count {cls_attn.shape[-1]} is not a perfect square")

        cls_attn = cls_attn.view(B, 1, grid, grid)
        cls_attn = F.interpolate(cls_attn, size=target_size, mode="bilinear", align_corners=False)
        mn = cls_attn.amin(dim=(1, 2, 3), keepdim=True)
        mx = cls_attn.amax(dim=(1, 2, 3), keepdim=True)
        return (cls_attn - mn) / (mx - mn + 1e-6)

    def cleanup(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class TextEmbedder:
    """Frozen CLIP text embedding (L2-normalised), shape (B, 512) for ViT-B/32."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def get_text_features(self, text_tokens: torch.Tensor) -> torch.Tensor:
        emb = self.model.encode_text(text_tokens.to(self.device)).float()
        return emb / emb.norm(dim=-1, keepdim=True)

    def cleanup(self):
        pass


class TextAttentionExtractor:
    """Legacy (v1) text features: mean CLS-row attention of the last N text blocks, (B, N)."""

    def __init__(self, model, device: str = "cpu", num_layers_to_extract: int = 3):
        self.model = model
        self.device = device
        self.num_layers = num_layers_to_extract
        self.text_attn_maps: list[torch.Tensor] = []
        self._hooks: list = []
        for layer in self.model.transformer.resblocks[-self.num_layers:]:
            self._hooks.append(layer.register_forward_hook(self._hook_fn))

    def _hook_fn(self, module, inputs, output):
        self.text_attn_maps.append(_attention_from_block(module, inputs[0], use_layernorm=False).detach())

    @torch.no_grad()
    def get_text_features(self, text_tokens: torch.Tensor) -> torch.Tensor:
        self.text_attn_maps.clear()
        self.model.encode_text(text_tokens.to(self.device))
        feats = [a[:, :, 0, 1:].mean(1).mean(-1, keepdim=True) for a in self.text_attn_maps]
        return torch.cat(feats, dim=-1).float()

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
