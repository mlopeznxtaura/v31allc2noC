"""RL transition schema for V31 training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Transition:
    """
    One training step. Implement V31DatasetBase to yield these from your data.

    Required for policy learning:
      - forward-pass inputs (hash, geometry, language)
      - action_token_id, reward

    Optional supervised targets (provide when available):
      - value_target / return_target — critic regression
      - next_frame_target — forward-model target (tri_emb or next_frame dim)
      - advantage_target — dueling head alignment
    """

    hash_str: str
    mission_id: int
    is_logical: bool
    uptime_sec: float
    step_idx: int
    is_sandbox: bool
    telemetry_stable: float
    system_id: str
    text_emb: torch.Tensor
    action_token_id: int
    reward: float
    done: bool = False
    value_target: Optional[float] = None
    return_target: Optional[float] = None
    next_frame_target: Optional[torch.Tensor] = None
    advantage_target: Optional[float] = None
    temperature: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def text_emb_2d(self) -> torch.Tensor:
        t = self.text_emb
        if t.dim() == 1:
            return t.unsqueeze(0)
        return t


def collate_transitions(batch: List[Transition]) -> Dict[str, Any]:
    """Stack a list of transitions for batched loss computation."""
    text_embs = torch.stack([t.text_emb_2d().squeeze(0) for t in batch])
    next_targets = None
    if all(t.next_frame_target is not None for t in batch):
        next_targets = torch.stack(
            [t.next_frame_target.squeeze() if t.next_frame_target.dim() > 0 else t.next_frame_target
             for t in batch]
        )
        if next_targets.dim() == 1:
            next_targets = next_targets.unsqueeze(0)

    value_targets = None
    if all(t.value_target is not None for t in batch):
        value_targets = torch.tensor([t.value_target for t in batch], dtype=torch.float32)

    return_targets = None
    if all(t.return_target is not None for t in batch):
        return_targets = torch.tensor([t.return_target for t in batch], dtype=torch.float32)

    adv_targets = None
    if all(t.advantage_target is not None for t in batch):
        adv_targets = torch.tensor([t.advantage_target for t in batch], dtype=torch.float32)

    return {
        "transitions": batch,
        "text_emb": text_embs,
        "action_token_id": torch.tensor([t.action_token_id for t in batch], dtype=torch.long),
        "reward": torch.tensor([t.reward for t in batch], dtype=torch.float32),
        "done": torch.tensor([t.done for t in batch], dtype=torch.float32),
        "next_frame_target": next_targets,
        "value_target": value_targets,
        "return_target": return_targets,
        "advantage_target": adv_targets,
    }
