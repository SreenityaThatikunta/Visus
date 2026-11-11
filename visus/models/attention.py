"""
Hooks into CLIP's transformer blocks to extract attention maps.
"""
import numpy as np
import torch
import torch.nn.functional as F


class AttentionExtractor:
    """Extracts spatial attention maps from the last visual transformer block."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.attn_maps: list[torch.Tensor] = []
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        last_block = self.model.visual.transformer.resblocks[-1]

        def _hook_fn(module, input, output):
            x = input[0]
            B = x.shape[1]
            qkv = F.linear(x, module.attn.in_proj_weight, module.attn.in_proj_bias)
            q, k, _ = qkv.chunk(3, dim=-1)

            num_heads = module.attn.num_heads
            head_dim  = q.shape[-1] // num_heads

            q = q.view(q.shape[0], B, num_heads, head_dim).permute(1, 2, 0, 3)
            k = k.view(k.shape[0], B, num_heads, head_dim).permute(1, 2, 0, 3)

            attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            self.attn_maps.append(attn.detach())

        self._hook = last_block.register_forward_hook(_hook_fn)

    def get_attention_map(
        self,
        image: torch.Tensor,
        text: torch.Tensor | None = None,
        target_size: tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        self.attn_maps.clear()
        image = image.to(self.device)

        with torch.no_grad():
            self.model.encode_image(image)
            if text is not None:
                self.model.encode_text(text.to(self.device))

        if not self.attn_maps:
            raise RuntimeError("No attention maps captured!")

        attn = self.attn_maps[-1]           # (B, H, L, L)
        B, _, _, _ = attn.shape
        cls_attn = attn[:, :, 0, 1:].mean(1)  # (B, num_patches)

        num_patches = cls_attn.shape[-1]
        grid = int(np.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a perfect square")

        cls_attn = cls_attn.view(B, 1, grid, grid)
        cls_attn = F.interpolate(cls_attn, size=target_size, mode="bilinear", align_corners=False)

        # Normalise to [0, 1]
        mn = cls_attn.view(B, -1).min(1)[0].view(B, 1, 1, 1)
        mx = cls_attn.view(B, -1).max(1)[0].view(B, 1, 1, 1)
        cls_attn = (cls_attn - mn) / (mx - mn + 1e-6)

        return cls_attn

    def cleanup(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class TextAttentionExtractor:
    """Extracts aggregated attention features from the last N text transformer blocks."""

    def __init__(self, model, device: str = "cpu", num_layers_to_extract: int = 3):
        self.model = model
        self.device = device
        self.num_layers = num_layers_to_extract
        self.text_attn_maps: list[dict] = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx, layer in enumerate(self.model.transformer.resblocks[-self.num_layers:]):
            def _hook_fn(module, input, output, layer_idx=layer_idx):
                x = input[0]
                B = x.shape[1]
                qkv = F.linear(x, module.attn.in_proj_weight, module.attn.in_proj_bias)
                q, k, _ = qkv.chunk(3, dim=-1)

                num_heads = module.attn.num_heads
                head_dim  = q.shape[-1] // num_heads

                q = q.view(q.shape[0], B, num_heads, head_dim).permute(1, 2, 0, 3)
                k = k.view(k.shape[0], B, num_heads, head_dim).permute(1, 2, 0, 3)

                attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn = attn.softmax(dim=-1)
                self.text_attn_maps.append({"layer_idx": layer_idx, "attention": attn.detach()})

            self._hooks.append(layer.register_forward_hook(_hook_fn))

    def get_text_features(self, text_tokens: torch.Tensor) -> torch.Tensor | None:
        self.text_attn_maps.clear()
        with torch.no_grad():
            self.model.encode_text(text_tokens.to(self.device))

        if not self.text_attn_maps:
            return None

        aggregated = []
        for layer_data in self.text_attn_maps:
            attn = layer_data["attention"]           # (B, H, T, T)
            cls_to_tokens = attn[:, :, 0, 1:]        # (B, H, T-1)
            cls_to_tokens = cls_to_tokens.mean(1)    # (B, T-1)
            global_feat   = cls_to_tokens.mean(-1, keepdim=True)  # (B, 1)
            aggregated.append(global_feat)

        return torch.cat(aggregated, dim=-1)          # (B, num_layers)

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
