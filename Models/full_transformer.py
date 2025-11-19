from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullTransformer(nn.Module):
    """
    Single-head attention + GELU MLP with a self-inclusive causal window.
    The block uses pre-LN, residual connections, and projects through a hidden width
    before returning to the original model dimension.
    """

    def __init__(
        self,
        *,
        dim: int,
        History: int,
        hidden_dim: int,
        device: Optional[torch.device] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.history = int(History)
        self.device = device
        self.hidden_dim = int(hidden_dim)

        self.in_proj = nn.Linear(dim, hidden_dim, bias=True, device=device)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True, device=device)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=True, device=device)

        self.ln_attn = nn.LayerNorm(hidden_dim, elementwise_affine=True, device=device)
        self.ln_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=True, device=device)
        self.drop = nn.Dropout(dropout)
        self.mlp_fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True, device=device)
        self.mlp_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True, device=device)
        self.mlp_act = nn.GELU()

        self._mask_cache: dict[Tuple[int, torch.device], torch.Tensor] = {}

    @torch.no_grad()
    def _attention_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = (T, device)
        if key not in self._mask_cache:
            idx = torch.arange(T, device=device)
            delta = idx[:, None] - idx[None, :]
            self._mask_cache[key] = (delta >= 0) & (delta <= self.history)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
        Returns:
            (B, T, dim)
        """
        if self.device is not None:
            x = x.to(self.device)
        B, T, _ = x.shape

        h = self.in_proj(x)

        # Attention block (pre-LN)
        norm_attn = self.ln_attn(h)
        q, k, v = self.qkv(norm_attn).chunk(3, dim=-1)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        if self.history >= 0:
            mask = self._attention_mask(T, x.device).unsqueeze(0).unsqueeze(0)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False).squeeze(1)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(1)

        h = h + self.drop(attn_out)

        # Feed-forward block (pre-LN)
        m = self.ln_mlp(h)
        m = self.mlp_fc2(self.mlp_act(self.mlp_fc1(m)))
        h = h + self.drop(m)

        return self.out_proj(h)
