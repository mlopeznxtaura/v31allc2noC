"""
v31/core/outputs.py — Two Output Tokens

NextFramePrediction:  What happens next based on all 7 inputs (Triangulation output)
LanguageTokenOutput:  RL-applied output token — the actual decision/action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NextFramePredictor(nn.Module):
    """
    Given triangulated state (B, tri_dim) + value_fn (B, 1),
    predict the next system state embedding.

    This is a forward model: s_{t+1} = f(tri_emb_t, V_t)
    Used for planning and lookahead within M3's time budget.
    """

    def __init__(self, tri_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tri_dim + 1, tri_dim),
            nn.GELU(),
            nn.LayerNorm(tri_dim),
            nn.Linear(tri_dim, out_dim),
            nn.Tanh(),  # bounded next-frame space
        )
        # Residual scale: small init to start near identity
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, tri_emb: torch.Tensor, value_fn: torch.Tensor) -> torch.Tensor:
        """
        tri_emb: (B, tri_dim)
        value_fn: (B, 1)
        Returns: next_frame (B, out_dim) — predicted next state
        """
        inp = torch.cat([tri_emb, value_fn], dim=-1)
        delta = self.net(inp)
        # Residual: next ≈ current + small delta (stability)
        if delta.shape == tri_emb.shape:
            return tri_emb + self.residual_scale * delta
        return delta


class LanguageTokenOutput(nn.Module):
    """
    RL-applied output token — the actual decision/action.

    Takes tri_emb + next_frame prediction → produces:
      - token_logits over vocab (the action distribution)
      - action_emb (continuous action embedding for policy gradient)
      - advantage estimate A(s,a) = Q(s,a) - V(s)

    This is the policy head π(a|s).
    """

    def __init__(self, tri_dim: int = 256, next_frame_dim: int = 256,
                 vocab_size: int = 32768, action_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size

        # Policy MLP
        self.policy_net = nn.Sequential(
            nn.Linear(tri_dim + next_frame_dim, action_dim),
            nn.GELU(),
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, action_dim),
            nn.GELU(),
        )

        # Discrete token head (for language output)
        self.token_head = nn.Linear(action_dim, vocab_size)

        # Continuous action embedding (for RL policy gradient)
        self.action_emb_head = nn.Linear(action_dim, action_dim)

        # Advantage head A(s,a) — used with value_fn to form Q
        self.advantage_head = nn.Linear(action_dim, 1)

    def forward(self, tri_emb: torch.Tensor,
                next_frame: torch.Tensor,
                value_fn: torch.Tensor,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          token_logits  (B, vocab_size)  — discrete token distribution
          action_emb    (B, action_dim)  — continuous action embedding
          advantage     (B, 1)           — A(s,a) estimate
          q_value       (B, 1)           — Q(s,a) = V(s) + A(s,a)
        """
        combined = torch.cat([tri_emb, next_frame], dim=-1)
        h = self.policy_net(combined)

        token_logits = self.token_head(h) / max(temperature, 1e-6)
        action_emb = self.action_emb_head(h)
        advantage = self.advantage_head(h)
        q_value = value_fn + advantage  # Dueling DQN-style

        return token_logits, action_emb, advantage, q_value
