"""
Pluggable dataset interface for V31 from-scratch training.

Implement V31DatasetBase (or provide get_dataset() in a Python module) and pass
--dataset-module path/to/your_dataset.py to train.py.

Example user module (your_dataset.py):

    from dataset import V31DatasetBase
    from transitions import Transition
    from embeddings import embed_text_simple

    class MyDataset(V31DatasetBase):
        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx: int) -> Transition:
            row = self.rows[idx]
            return Transition(
                hash_str=row["hash"],
                mission_id=row["mission_id"],
                is_logical=True,
                uptime_sec=row["uptime"],
                step_idx=row["step"],
                is_sandbox=True,
                telemetry_stable=1.0,
                system_id="train",
                text_emb=embed_text_simple(row["text"], dim=128),
                action_token_id=row["action"],
                reward=row["reward"],
                done=row["done"],
                value_target=row.get("value"),
                next_frame_target=row.get("next_emb"),
            )

    def get_dataset():
        return MyDataset()
"""

import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import torch

from transitions import Transition


class V31DatasetBase(ABC):
    """Yield Transition objects — one RL step per sample."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Transition:
        raise NotImplementedError

    def episodes(self) -> List[List[Transition]]:
        """Optional: group into episodes (reset boundaries via done=True)."""
        episodes: List[List[Transition]] = []
        current: List[Transition] = []
        for i in range(len(self)):
            t = self[i]
            current.append(t)
            if t.done:
                episodes.append(current)
                current = []
        if current:
            episodes.append(current)
        return episodes


def load_dataset_from_module(module_path: str) -> V31DatasetBase:
    path = Path(module_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset module not found: {path}")

    spec = importlib.util.spec_from_file_location("v31_user_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module: {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "get_dataset"):
        ds = mod.get_dataset()
    elif hasattr(mod, "DATASET"):
        ds = mod.DATASET
    else:
        raise ValueError(
            f"{path} must define get_dataset() or DATASET (a V31DatasetBase instance)"
        )

    if not isinstance(ds, V31DatasetBase):
        raise TypeError("Dataset must subclass V31DatasetBase")
    return ds


class SyntheticSmokeDataset(V31DatasetBase):
    """Random transitions for pipeline smoke tests only — not real training data."""

    def __init__(self, n: int = 64, embed_dim: int = 128, next_frame_dim: int = 256):
        import random

        from embeddings import embed_text_simple

        self._items: List[Transition] = []
        for i in range(n):
            text = f"synthetic step {i} mission active"
            self._items.append(
                Transition(
                    hash_str=f"smoke-{i % 8}",
                    mission_id=1,
                    is_logical=True,
                    uptime_sec=float(i * 10),
                    step_idx=i % 8,
                    is_sandbox=True,
                    telemetry_stable=1.0,
                    system_id="smoke",
                    text_emb=embed_text_simple(text, dim=embed_dim),
                    action_token_id=random.randint(0, 1023),
                    reward=random.random(),
                    done=(i % 8 == 7),
                    value_target=random.random(),
                    next_frame_target=torch.randn(next_frame_dim),
                )
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Transition:
        return self._items[idx]
