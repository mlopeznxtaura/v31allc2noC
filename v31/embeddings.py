"""Shared text embedding utilities for V31 inference and training."""

import hashlib
import torch


def embed_text_simple(text: str, dim: int = 128) -> torch.Tensor:
    """Minimal text → vector without external deps. Hash-based embedding."""
    tokens = text.lower().split()
    vecs = []
    for tok in tokens[:dim // 8]:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        bits = [(h >> i) & 1 for i in range(16)]
        vecs.append(bits)
    while len(vecs) < dim // 16:
        vecs.append([0] * 16)
    flat = [b for v in vecs for b in v][:dim]
    while len(flat) < dim:
        flat.append(0)
    return torch.tensor([flat], dtype=torch.float32)
