"""Checkpoint save/load for V31 training and deployment."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    epoch: int = 0,
    config: Optional[dict] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, out)
    # Deployment-friendly weights-only copy
    weights_path = out.with_suffix(".pt")
    if weights_path != out:
        torch.save(model.state_dict(), weights_path)


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Tuple[int, int, dict]:
    """Load full training checkpoint. Returns (step, epoch, payload)."""
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return (
            int(payload.get("step", 0)),
            int(payload.get("epoch", 0)),
            payload,
        )
    # weights-only file
    model.load_state_dict(payload)
    return 0, 0, {"model_state_dict": payload}


def load_weights_only(path: str, model: nn.Module, map_location: str = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def write_training_manifest(checkpoint_dir: str, config: dict) -> None:
    manifest = {
        "format": "v31-checkpoint",
        "config": config,
        "files": {
            "latest": "checkpoint_latest.pth",
            "weights": "checkpoint_latest.pt",
        },
    }
    path = Path(checkpoint_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
