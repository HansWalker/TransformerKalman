from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformer(nn.Module):
    """
    Single-head self-attention with an optional H-step causal window (inclusive).
    Token i attends to j iff 0 <= i - j <= History.  If History < 0 the module
    falls back to the standard causal mask.
    """

    def __init__(
        self,
        *,
        dim: int,
        History: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.history = int(History)
        self.device = device

        self.qkv = (
            nn.Linear(dim, 3 * dim, bias=True, device=device)
            if device is not None
            else nn.Linear(dim, 3 * dim, bias=True)
        )
        self._mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    @torch.no_grad()
    def _attention_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        if key not in self._mask_cache:
            idx = torch.arange(T, device=device)
            i = idx[:, None]
            j = idx[None, :]
            delta = i - j
            mask = (delta >= 0) & (delta <= self.history)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) context from the causal windowed SDPA.
        """
        if self.device is not None:
            x = x.to(self.device)
        B, T, _ = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # (B, T, D) each
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        if self.history >= 0:
            mask = self._attention_mask(T, x.device).unsqueeze(0).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=False
            ).squeeze(1)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(1)
        return attn_out
