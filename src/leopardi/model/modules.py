"""Core building blocks for Leopardi, using modern transformer components.

All modules are torch.compile-friendly: no Python-level branching on tensor
values in forward(), standard ops throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# RMSNorm (faster and stabler than LayerNorm, per Qwen3 / LLaMA)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embeddings, per Qwen3 / LLaMA)
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tensor:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = pos.unsqueeze(1) / (theta ** (dim.unsqueeze(0) / head_dim))
    return torch.stack((freqs.cos(), freqs.sin()), dim=-1)  # (seq, head_dim/2, 2)


def apply_rope(x: Tensor, rope_cache: Tensor) -> Tensor:
    """Apply rotary embeddings. x: (batch, heads, seq, head_dim)."""
    seq_len = x.size(2)
    rc = rope_cache[:seq_len]  # (seq, head_dim/2, 2)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., head_dim/2, 2)
    cos = rc[..., 0].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim/2)
    sin = rc[..., 1].unsqueeze(0).unsqueeze(0)
    x0, x1 = x_pairs[..., 0], x_pairs[..., 1]
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape(x.shape).to(x.dtype)


# ---------------------------------------------------------------------------
# SwiGLU FFN (per Qwen3 / LLaMA / PaLM)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 2.67, dropout: float = 0.1) -> None:
        super().__init__()
        intermediate = int(hidden_size * mlp_ratio)
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# GQA Self-Attention (Grouped Query Attention, per Qwen3)
# ---------------------------------------------------------------------------

class GQASelfAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, rope_cache: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_cache is not None:
            q = apply_rope(q, rope_cache)
            k = apply_rope(k, rope_cache)
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# GQA Cross-Attention
# ---------------------------------------------------------------------------

class GQACrossAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        B, S, _ = x.shape
        _, T, _ = context.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Transformer blocks using modern components
# ---------------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """Cross-attention block for the structural latent bottleneck."""

    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        mlp_ratio: float = 2.67, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(hidden_size)
        self.self_attn = GQASelfAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.cross_norm = RMSNorm(hidden_size)
        self.cross_attn = GQACrossAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, mlp_ratio, dropout)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_norm(x))
        x = x + self.cross_attn(self.cross_norm(x), context)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PlannerBlock(nn.Module):
    """Query-to-latent cross-attention block for the block planner."""

    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        mlp_ratio: float = 2.67, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query_norm = RMSNorm(hidden_size)
        self.query_attn = GQASelfAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.cross_norm = RMSNorm(hidden_size)
        self.cross_attn = GQACrossAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, mlp_ratio, dropout)

    def forward(self, queries: Tensor, latents: Tensor) -> Tensor:
        queries = queries + self.query_attn(self.query_norm(queries))
        queries = queries + self.cross_attn(self.cross_norm(queries), latents)
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class WriterBlock(nn.Module):
    """Causal self-attention + cross-attention to memory + SwiGLU FFN."""

    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int,
        mlp_ratio: float = 2.67, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(hidden_size)
        self.self_attn = GQASelfAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.cross_norm = RMSNorm(hidden_size)
        self.cross_attn = GQACrossAttention(hidden_size, num_heads, num_kv_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, mlp_ratio, dropout)

    def forward(
        self, x: Tensor, memory: Tensor,
        rope_cache: Tensor | None = None, causal_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.self_norm(x), rope_cache=rope_cache, mask=causal_mask)
        x = x + self.cross_attn(self.cross_norm(x), memory)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Page canonicalizer (non-neural, same as before)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CanonicalizedPage:
    image: Tensor
    line_density_map: Tensor
    text_direction_map: Tensor


class PageCanonicalizer(nn.Module):
    def __init__(self, line_density_pool: int = 16, contrast_eps: float = 1e-5) -> None:
        super().__init__()
        self.line_density_pool = line_density_pool
        self.contrast_eps = contrast_eps

    def forward(self, image: Tensor) -> CanonicalizedPage:
        if image.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        mean = image.mean(dim=(-2, -1), keepdim=True)
        std = image.std(dim=(-2, -1), keepdim=True).clamp_min(self.contrast_eps)
        normalized = (image - mean) / std
        gray = normalized.mean(dim=1, keepdim=True)
        density = F.avg_pool2d(gray.abs(), kernel_size=self.line_density_pool, stride=self.line_density_pool)
        grad_x = gray[..., :, 1:] - gray[..., :, :-1]
        grad_y = gray[..., 1:, :] - gray[..., :-1, :]
        x_dir = F.pad(grad_x.abs(), (0, 1, 0, 0))
        y_dir = F.pad(grad_y.abs(), (0, 0, 0, 1))
        text_direction_map = torch.cat((x_dir, y_dir), dim=1)
        return CanonicalizedPage(image=normalized, line_density_map=density, text_direction_map=text_direction_map)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def pixel_shuffle_down(x: Tensor, factor: int = 2) -> Tensor:
    """Merge spatial patches: (B, C, H, W) -> (B, C*factor^2, H//factor, W//factor)."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * factor * factor, H // factor, W // factor)
    return x


def flatten_pooled_feature_map(feature_map: Tensor, grid: Iterable[int]) -> Tensor:
    h, w = [int(v) for v in grid]
    pooled = F.adaptive_avg_pool2d(feature_map, output_size=(h, w))
    return pooled.flatten(2).transpose(1, 2)
