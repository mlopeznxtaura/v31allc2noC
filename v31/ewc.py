"""Elastic Weight Consolidation for continual V31 training."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class EWCPenalty:
    """
    Diagonal Fisher EWC penalty: λ Σ F_i (θ_i - θ*_i)²
    Call anchor() after a training phase; penalty() during later phases.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0):
        self.ewc_lambda = ewc_lambda
        self._anchor: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self._model = model

    def is_active(self) -> bool:
        return bool(self._anchor)

    def anchor(self, fisher: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Store current weights as θ*. Optionally set Fisher diagonal."""
        self._anchor = {k: p.detach().clone() for k, p in self._model.named_parameters()}
        if fisher is not None:
            self._fisher = fisher
        else:
            self._fisher = {k: torch.ones_like(p) for k, p in self._anchor.items()}

    def accumulate_fisher(self, loss: torch.Tensor) -> None:
        """One-step diagonal Fisher estimate from gradients of `loss`."""
        self._model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        for name, p in self._model.named_parameters():
            if p.grad is None:
                continue
            g2 = p.grad.detach().pow(2)
            if name not in self._fisher:
                self._fisher[name] = g2.clone()
            else:
                self._fisher[name] += g2

    def normalize_fisher(self, n_steps: int) -> None:
        if n_steps <= 0:
            return
        for name in self._fisher:
            self._fisher[name] /= float(n_steps)

    def penalty(self) -> torch.Tensor:
        if not self._anchor:
            return torch.tensor(0.0)
        total = torch.tensor(0.0, device=next(self._model.parameters()).device)
        for name, p in self._model.named_parameters():
            if name not in self._anchor:
                continue
            diff = p - self._anchor[name]
            fisher = self._fisher.get(name, torch.ones_like(p))
            total = total + (fisher * diff.pow(2)).sum()
        return self.ewc_lambda * total
