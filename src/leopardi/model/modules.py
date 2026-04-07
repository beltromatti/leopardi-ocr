from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


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


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pw2(x)
        return residual + x


class ConvStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim == out_dim:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            )
        self.blocks = nn.Sequential(*[ConvNeXtBlock(out_dim, dropout=dropout) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)[0]
        x = x + self.cross_attn(self.cross_norm(x), context, context, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class PlannerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, queries: Tensor, latents: Tensor) -> Tensor:
        queries = queries + self.query_attn(
            self.query_norm(queries),
            self.query_norm(queries),
            self.query_norm(queries),
            need_weights=False,
        )[0]
        queries = queries + self.cross_attn(
            self.cross_norm(queries), latents, latents, need_weights=False
        )[0]
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class WriterBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: Tensor, memory: Tensor, causal_mask: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(
            self.self_norm(x),
            self.self_norm(x),
            self.self_norm(x),
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        x = x + self.cross_attn(self.cross_norm(x), memory, memory, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


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
        if image.dtype != torch.float32 and image.dtype != torch.float16 and image.dtype != torch.bfloat16:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        mean = image.mean(dim=(-2, -1), keepdim=True)
        std = image.std(dim=(-2, -1), keepdim=True).clamp_min(self.contrast_eps)
        normalized = (image - mean) / std

        gray = normalized.mean(dim=1, keepdim=True)
        density = torch.nn.functional.avg_pool2d(
            gray.abs(), kernel_size=self.line_density_pool, stride=self.line_density_pool
        )
        grad_x = gray[..., :, 1:] - gray[..., :, :-1]
        grad_y = gray[..., 1:, :] - gray[..., :-1, :]
        x_dir = torch.nn.functional.pad(grad_x.abs(), (0, 1, 0, 0))
        y_dir = torch.nn.functional.pad(grad_y.abs(), (0, 0, 0, 1))
        text_direction_map = torch.cat((x_dir, y_dir), dim=1)
        return CanonicalizedPage(
            image=normalized,
            line_density_map=density,
            text_direction_map=text_direction_map,
        )


def make_causal_mask(length: int, device: torch.device) -> Tensor:
    mask = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def flatten_pooled_feature_map(feature_map: Tensor, grid: Iterable[int]) -> Tensor:
    h, w = [int(value) for value in grid]
    pooled = torch.nn.functional.adaptive_avg_pool2d(feature_map, output_size=(h, w))
    return pooled.flatten(2).transpose(1, 2)
