import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from origami_iros._typing import Action
from origami_iros.modules.base import BaseActionModule


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def sample_time_beta(
    batch_size: int, device: torch.device, alpha: float = 1.5, beta: float = 1.0,
    scale: float = 0.999, offset: float = 0.001,
) -> torch.Tensor:
    t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
    return t * scale + offset


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        x = x + self.attn(h, memory, memory, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CausalSelfAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        x = x + self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FlowMatchingActionHead(BaseActionModule[torch.Tensor]):
    def __init__(
        self,
        chunk_size: int = 13,
        action_dim: int = 65,
        dim_in: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_inference_steps: int = 10,
    ) -> None:
        super().__init__(dim_action=chunk_size * action_dim)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_inference_steps = num_inference_steps

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.obs_proj = nn.Linear(dim_in, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

        # alternates CA, SA, CA, SA, ...
        self.blocks = nn.ModuleList([
            CrossAttnBlock(hidden_dim, num_heads) if i % 2 == 0 else CausalSelfAttnBlock(hidden_dim, num_heads)
            for i in range(num_layers)
        ])

        causal_mask = torch.triu(torch.full((chunk_size, chunk_size), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def predict_velocity(self, noisy_action: torch.Tensor, t: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # noisy_action: (batch, chunk_size, action_dim), t: (batch,), memory: (batch, n_tokens, hidden_dim)
        act_emb = self.action_proj(noisy_action)
        t_emb = self.time_mlp(t).unsqueeze(1)
        x = act_emb + t_emb

        for block in self.blocks:
            x = block(x, memory) if isinstance(block, CrossAttnBlock) else block(x, self.causal_mask)

        return self.out_proj(x)  # (batch, chunk_size, action_dim)

    def sample_actions(
        self,
        obs_tokens: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        batch_size = obs_tokens.size(0)
        device, dtype = obs_tokens.device, obs_tokens.dtype
        num_steps = num_inference_steps or self.num_inference_steps

        memory = self.obs_proj(obs_tokens)

        act = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device, dtype=dtype) if noise is None else noise

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_curr = torch.full((batch_size,), i * dt, device=device, dtype=dtype)
            v_pred = self.predict_velocity(act, t_curr, memory)
            act = act + v_pred * dt

        return act.reshape(batch_size, self.dim_action)

    def forward(self, x: torch.Tensor) -> Action:
        return self.sample_actions(x)

    def compute_loss(
        self,
        x: torch.Tensor,
        target_action: torch.Tensor,
        noise: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        target_action = target_action.reshape(batch_size, self.chunk_size, self.action_dim)

        t = sample_time_beta(batch_size, device)
        x_0 = torch.randn_like(target_action) if noise is None else noise

        t_expanded = t[:, None, None]
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * target_action
        v_target = target_action - x_0

        memory = self.obs_proj(x)
        v_pred = self.predict_velocity(x_t, t, memory)

        if loss_mask is None:
            return F.mse_loss(v_pred, v_target)

        per_elem = F.mse_loss(v_pred, v_target, reduction="none")
        mask = (~loss_mask).float().unsqueeze(-1)
        return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)
