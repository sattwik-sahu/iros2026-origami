"""Time-conditioned transformer that predicts the flow-matching velocity field.

This module contains the transport-agnostic neural network that the
:mod:`origami_iros.models.action_head.fm` head uses as its *velocity model*.
It is intentionally kept free of any flow-matching machinery: it only maps an
``(noisy_action, time, observation_tokens)`` triple to a predicted velocity.
The actual flow-matching bookkeeping (probability path, scheduler, ODE solver)
lives in :mod:`origami_iros.models.action_head.fm` and is delegated to the
reference ``flow_matching`` library.
"""

from __future__ import annotations

import math

import torch
from typing import override
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding of a scalar (used for the time ``t``).

    Given a scalar (or a batch of scalars) ``x``, returns a vector of
    ``dim`` sinusoidal components typically summed into token embeddings to
    condition a transformer on the flow time ``t``.

    Attributes:
        dim: Output embedding dimension (rounded up to an even number).
    """

    def __init__(self, dim: int) -> None:
        """Initialise the embedding.

        Args:
            dim: Desired output dimension.
        """
        super().__init__()
        self.dim = dim

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar time values.

        Accepts either a scalar (0-dimensional) time, as passed by ODE solvers
        during integration, or a batch of times of shape ``(batch,)``.

        Args:
            x: Input time value(s); a scalar or shape ``(batch,)``.

        Returns:
            Sinusoidal embedding of shape ``(1, dim)`` for a scalar input or
            ``(batch, dim)`` for a batched input.
        """
        device = x.device
        if x.dim() == 0:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CrossAttnBlock(nn.Module):
    """A self-attention-free block that cross-attends over observation tokens.

    The module applies pre-norm multi-head cross-attention using the
    observation tokens as keys/values, followed by a two-layer MLP feed-forward
    network. This lets each action-timestep token read information from the
    encoded observation.

    Attributes:
        hidden_dim: Token/embedding dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        """Initialise the block.

        Args:
            hidden_dim: Embedding dimension of tokens.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    @override
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention and FFN with residual connections.

        Args:
            x: Query tokens of shape ``(batch, seq, hidden_dim)``.
            memory: Observation tokens of shape ``(batch, n_tokens, hidden_dim)``.

        Returns:
            Updated tokens of the same shape as ``x``.
        """
        h = self.norm(x)
        x = x + self.attn(h, memory, memory, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CausalSelfAttnBlock(nn.Module):
    """Causal self-attention block over the action chunk.

    Applies masked self-attention so that earlier action-timestep tokens cannot
    look ahead into future ones, followed by a feed-forward network. This enforces
    an autoregressive ordering over the action chunk.

    Attributes:
        hidden_dim: Token/embedding dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        """Initialise the block.

        Args:
            hidden_dim: Embedding dimension of tokens.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    @override
    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention and FFN with residual connections.

        Args:
            x: Action tokens of shape ``(batch, chunk_size, hidden_dim)``.
            causal_mask: Attention mask of shape ``(chunk_size, chunk_size)``.

        Returns:
            Updated tokens of the same shape as ``x``.
        """
        h = self.norm(x)
        x = x + self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ActionDiT(nn.Module):
    """Time-conditioned transformer velocity network for the action head.

    This is the neural backbone that receives a batch of noisy action chunks at a
    given flow time ``t`` and predicts the corresponding velocity (the vector
    pointing from the noisy sample towards the target action). It alternates
    cross-attention blocks (over the encoded observation) and causal self-attention
    blocks (over the action chunk), mirroring a diffusion-transformer (DiT) style
    body.

    Attributes:
        chunk_size: Number of action timesteps in a chunk.
        action_dim: Dimensionality of a single action vector.
        hidden_dim: Token/embedding dimension.
        num_layers: Number of alternating transformer blocks.
        num_heads: Number of attention heads in each block.
    """

    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        dim_in: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        """Initialise the velocity network.

        Args:
            chunk_size: Number of action timesteps in a chunk.
            action_dim: Dimensionality of a single action vector.
            dim_in: Dimension of the observation tokens feeding the head.
            hidden_dim: Embedding dimension for the transformer tokens.
            num_layers: Number of alternating cross/causal-attention blocks.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.obs_proj = nn.Linear(dim_in, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

        self.blocks = nn.ModuleList(
            [
                CrossAttnBlock(hidden_dim, num_heads)
                if i % 2 == 0
                else CausalSelfAttnBlock(hidden_dim, num_heads)
                for i in range(num_layers)
            ]
        )

        causal_mask = torch.triu(
            torch.full((chunk_size, chunk_size), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    @override
    def forward(
        self, noisy_action: torch.Tensor, t: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Predict the velocity for a batch of noisy action chunks.

        Args:
            noisy_action: Noisy action chunk of shape
                ``(batch, chunk_size, action_dim)``.
            t: Flow time for each batch element, shape ``(batch,)``.
            memory: Observation tokens of shape ``(batch, n_tokens, hidden_dim)``.

        Returns:
            Predicted velocity of shape ``(batch, chunk_size, action_dim)``.
        """
        act_emb = self.action_proj(noisy_action)
        t_emb = self.time_mlp(t).unsqueeze(1)
        x = act_emb + t_emb

        for block in self.blocks:
            if isinstance(block, CrossAttnBlock):
                x = block(x, memory)
            else:
                x = block(x, self.causal_mask)

        return self.out_proj(x)

    def memory_from_tokens(self, obs_tokens: torch.Tensor) -> torch.Tensor:
        """Project raw observation tokens into the transformer memory space.

        Args:
            obs_tokens: Encoded observation tokens of shape
                ``(batch, n_tokens, dim_in)``.

        Returns:
            Memory tokens of shape ``(batch, n_tokens, hidden_dim)``.
        """
        return self.obs_proj(obs_tokens)
