"""
v31/agents.py — M1, M2, M3 scalar agent definitions

M1: Unbounded thinker. Emotionally damaged neural net simulation.
    Explores forever, generates infinite logical branches, no internal valence signal.

M2: Efficiency zealot. Observes M1, enforces time-based rules.
    Strict/extreme efficiency, blind to novel discovery.

M3: Meta-observer. Observes M1 + M2. Determines efficiency/logic/creative multipliers.
    Manages its own time budget, can throttle or reparameterize both.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class M1_UnboundedThinker(nn.Module):
    """
    Emotionally damaged neural net — no valence gating.
    Produces a scalar embedding by exploring all branches without pruning.
    Branch count controlled externally by M3.
    """

    def __init__(self, d_in: int = 64, d_out: int = 64, max_branches: int = 64):
        super().__init__()
        self.max_branches = max_branches
        # Each branch is a small MLP; no softmax or selection — all pass through
        self.branch_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, d_out), nn.GELU())
            for _ in range(max_branches)
        ])
        self.collapse = nn.Linear(d_out * max_branches, d_out)
        self.active_branches = max_branches  # M3 can set this

    def set_branch_budget(self, n: int):
        """M3 throttle hook."""
        self.active_branches = max(1, min(n, self.max_branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_in) → scalar embedding (B, d_out)"""
        branches = [self.branch_projs[i](x) for i in range(self.active_branches)]
        # Pad inactive branches with zeros to keep shape stable
        if self.active_branches < self.max_branches:
            pad = torch.zeros(x.shape[0], (self.max_branches - self.active_branches) * branches[0].shape[-1],
                              device=x.device)
            cat = torch.cat([torch.cat(branches, dim=-1), pad], dim=-1)
        else:
            cat = torch.cat(branches, dim=-1)
        return self.collapse(cat)  # (B, d_out)


class M2_EfficiencyZealot(nn.Module):
    """
    Observes M1 output + raw input. Enforces time-step rules.
    Returns a gating scalar that suppresses M1 excess.
    Blind to novelty — will kill discovery branches by design.
    """

    def __init__(self, d_in: int = 64, d_out: int = 64, max_steps: int = 8):
        super().__init__()
        self.max_steps = max_steps
        self.step_counter = 0
        # Strict linear projection — no residual, no exploration
        self.proj = nn.Sequential(
            nn.Linear(d_in * 2, d_out),  # [raw_input | m1_output]
            nn.ReLU(),                    # Hard activation — no soft paths
            nn.Linear(d_out, d_out),
        )
        self.gate = nn.Linear(d_out, 1)  # Scalar gate ∈ [0,1]

    def reset_step(self):
        self.step_counter = 0

    def forward(self, raw: torch.Tensor, m1_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        raw: (B, d_in), m1_out: (B, d_in)
        Returns: m2_embedding (B, d_out), efficiency_gate (B, 1)
        """
        self.step_counter += 1
        # After max_steps, gate collapses to zero (hard kill)
        if self.step_counter > self.max_steps:
            zero_emb = torch.zeros(raw.shape[0], self.proj[-1].out_features, device=raw.device)
            zero_gate = torch.zeros(raw.shape[0], 1, device=raw.device)
            return zero_emb, zero_gate

        combined = torch.cat([raw, m1_out], dim=-1)
        emb = self.proj(combined)
        gate = torch.sigmoid(self.gate(emb))
        return emb, gate


class M3_MetaObserver(nn.Module):
    """
    Observes M1 + M2 outputs. Computes efficiency, logic, creative multipliers.
    Manages its own time budget. Can throttle or reparameterize M1 and M2.
    Returns M3 scalar embedding + reparameterization signals.
    """

    def __init__(self, d_m1: int = 64, d_m2: int = 64, d_out: int = 64, repar_interval: int = 4):
        super().__init__()
        self.repar_interval = repar_interval
        self._cycle = 0
        self._time_budget = 1.0  # [0,1] self-managed

        self.obs_proj = nn.Linear(d_m1 + d_m2, d_out)

        # Three multiplier heads
        self.efficiency_head = nn.Linear(d_out, 1)   # M2 multiplier
        self.logic_head = nn.Linear(d_out, 1)         # M1 logic multiplier
        self.creative_head = nn.Linear(d_out, 1)      # M1 branch multiplier

        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, m1_emb: torch.Tensor, m2_emb: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Returns: m3_emb (B, d_out), efficiency_mult, logic_mult, creative_mult — each (B,1)
        """
        self._cycle += 1
        combined = torch.cat([m1_emb, m2_emb], dim=-1)
        h = torch.relu(self.obs_proj(combined))

        efficiency_mult = torch.sigmoid(self.efficiency_head(h))  # [0,1]
        logic_mult = torch.sigmoid(self.logic_head(h))
        creative_mult = torch.sigmoid(self.creative_head(h))

        # Self-managed time budget: decay over cycles
        self._time_budget = max(0.05, self._time_budget * 0.99)

        m3_emb = self.out_proj(h * self._time_budget)
        return m3_emb, efficiency_mult, logic_mult, creative_mult

    def reparameterize(self, m1: M1_UnboundedThinker, m2: M2_EfficiencyZealot,
                       creative_mult: torch.Tensor, efficiency_mult: torch.Tensor):
        """Apply multipliers back to M1/M2 if at repar interval."""
        if self._cycle % self.repar_interval == 0:
            cm = creative_mult.mean().item()
            new_branches = max(1, int(m1.max_branches * cm))
            m1.set_branch_budget(new_branches)

            em = efficiency_mult.mean().item()
            new_steps = max(1, int(m2.max_steps * em))
            m2.max_steps = new_steps
