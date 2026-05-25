"""
v31/core/scalars.py — Four Input Scalars

Binary:        Raw machine initialization (codebase hash, mission, purpose, is it logical?)
Geometry:      Where am I in space/time? Telemetry stable? Sandbox vs live? System ID?
Language:      Rich contextualization of Binary + Geometry with narrative
Triangulation: RL value function from all 6 inputs (M1+M2+M3+Binary+Geometry+Language)
               Receives 7 total (including itself as prior), outputs value function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import time
import json
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Scalar 4: Binary
# ─────────────────────────────────────────────────────────────────────────────
class BinaryScalar(nn.Module):
    """
    Encodes raw machine state:
      - codebase hash (deterministic from source)
      - mission/purpose flag
      - logical consistency check

    Input: dict with hash_str, mission_id (int), is_logical (bool)
    Output: (B, binary_dim)
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.hash_proj = nn.Linear(64, dim // 2)
        self.meta_proj = nn.Linear(4, dim // 2)    # [mission_id, is_logical, timestamp_norm, entropy]
        self.fuse = nn.Linear(dim, dim)

    @staticmethod
    def hash_to_vec(hash_str: str, length: int = 64) -> torch.Tensor:
        """Convert hex hash string to float vector via bit decomposition."""
        h = hashlib.sha256(hash_str.encode()).hexdigest()
        bits = []
        for ch in h[:length // 4]:
            val = int(ch, 16)
            bits.extend([(val >> i) & 1 for i in range(4)])
        bits = bits[:length]
        while len(bits) < length:
            bits.append(0)
        return torch.tensor(bits, dtype=torch.float32)

    def forward(self, hash_str: str, mission_id: int, is_logical: bool,
                batch_size: int = 1) -> torch.Tensor:
        h_vec = self.hash_to_vec(hash_str).unsqueeze(0).expand(batch_size, -1)  # (B, 64)
        ts_norm = (time.time() % 86400) / 86400.0
        entropy = float(len(set(hash_str))) / 16.0  # hex diversity [0,1]
        meta = torch.tensor(
            [[mission_id / 255.0, float(is_logical), ts_norm, entropy]],
            dtype=torch.float32
        ).expand(batch_size, -1)  # (B, 4)

        h_vec = h_vec.to(next(self.parameters()).device)
        meta = meta.to(next(self.parameters()).device)

        h_emb = F.gelu(self.hash_proj(h_vec))   # (B, dim//2)
        m_emb = F.gelu(self.meta_proj(meta))     # (B, dim//2)
        fused = torch.cat([h_emb, m_emb], dim=-1)
        return self.fuse(fused)  # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar 5: Geometry
# ─────────────────────────────────────────────────────────────────────────────
class GeometryScalar(nn.Module):
    """
    Spatial/temporal self-location:
      - uptime, step index (temporal)
      - sandbox vs live flag
      - telemetry stability score
      - system_id hash

    Output: (B, geo_dim)
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(8, dim)
        self.pos_enc = nn.Parameter(torch.randn(1, dim) * 0.02)

    def forward(self, uptime_sec: float, step_idx: int, is_sandbox: bool,
                telemetry_stable: float, system_id: str,
                batch_size: int = 1) -> torch.Tensor:
        # Encode system_id as entropy scalar
        sys_entropy = len(set(system_id)) / max(len(system_id), 1)
        sys_hash_val = (int(hashlib.md5(system_id.encode()).hexdigest()[:8], 16) % 1000) / 1000.0

        features = torch.tensor([[
            min(uptime_sec / 86400.0, 1.0),           # uptime (normalized to day)
            math.log1p(step_idx) / 20.0,               # step index (log scale)
            float(is_sandbox),                          # sandbox flag
            float(not is_sandbox),                      # live flag
            min(telemetry_stable, 1.0),                 # stability [0,1]
            1.0 - min(telemetry_stable, 1.0),           # instability
            sys_entropy,                                # system id diversity
            sys_hash_val,                               # system id hash fingerprint
        ]], dtype=torch.float32).expand(batch_size, -1)

        features = features.to(next(self.parameters()).device)
        emb = F.gelu(self.proj(features))
        return emb + self.pos_enc  # (B, dim)


import math  # noqa: E402 — placed here to avoid circular at top


# ─────────────────────────────────────────────────────────────────────────────
# Scalar 6: Language
# ─────────────────────────────────────────────────────────────────────────────
class LanguageScalar(nn.Module):
    """
    Rich narrative contextualization of Binary + Geometry.
    Takes token ids or pre-embedded text vector and fuses with
    binary/geometry context to produce language scalar.

    Input: pre-embedded vector (B, embed_dim) — use any tokenizer upstream
    Output: (B, lang_dim)
    """

    def __init__(self, embed_dim: int = 128, binary_dim: int = 128,
                 geo_dim: int = 64, lang_dim: int = 128):
        super().__init__()
        self.text_proj = nn.Linear(embed_dim, lang_dim // 2)
        self.ctx_proj = nn.Linear(binary_dim + geo_dim, lang_dim // 2)
        self.fuse = nn.Sequential(
            nn.Linear(lang_dim, lang_dim),
            nn.GELU(),
            nn.LayerNorm(lang_dim),
        )

    def forward(self, text_emb: torch.Tensor,
                binary_emb: torch.Tensor,
                geo_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb: (B, embed_dim)
        binary_emb: (B, binary_dim)
        geo_emb: (B, geo_dim)
        Returns: (B, lang_dim)
        """
        t = F.gelu(self.text_proj(text_emb))
        ctx = torch.cat([binary_emb, geo_emb], dim=-1)
        c = F.gelu(self.ctx_proj(ctx))
        return self.fuse(torch.cat([t, c], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Scalar 7: Triangulation  ← receives ALL 6 inputs + itself = 7 total
# ─────────────────────────────────────────────────────────────────────────────
class TriangulationScalar(nn.Module):
    """
    RL value function triangulating across all 7 inputs:
      M1, M2, M3, Binary, Geometry, Language + self (prior triangulation state)

    Invariant: output magnitude bounded, value function monotone in quality.

    Outputs:
      tri_emb       (B, tri_dim)  — triangulated state vector
      value_fn      (B, 1)        — RL state value estimate V(s)
    """

    def __init__(self, m1_dim: int = 64, m2_dim: int = 64, m3_dim: int = 64,
                 binary_dim: int = 128, geo_dim: int = 64, lang_dim: int = 128,
                 tri_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.tri_dim = tri_dim
        total_in = m1_dim + m2_dim + m3_dim + binary_dim + geo_dim + lang_dim
        self_in = tri_dim  # prior triangulation

        # Project each stream to tri_dim for attention
        self.stream_projs = nn.ModuleList([
            nn.Linear(d, tri_dim)
            for d in [m1_dim, m2_dim, m3_dim, binary_dim, geo_dim, lang_dim, tri_dim]
        ])  # 7 projections

        # Cross-stream multi-head attention over the 7 tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=tri_dim, num_heads=n_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(tri_dim)

        # Value function MLP — invariant-preserving (monotone sigmoid output)
        self.value_mlp = nn.Sequential(
            nn.Linear(tri_dim, tri_dim // 2),
            nn.GELU(),
            nn.Linear(tri_dim // 2, 1),
            nn.Sigmoid(),   # V(s) ∈ [0,1]
        )

        # Output projection preserving magnitude
        self.out_proj = nn.Linear(tri_dim, tri_dim)
        self.ln2 = nn.LayerNorm(tri_dim)

        # Invariant: register unit sphere normalizer
        self._invariant_scale = nn.Parameter(torch.ones(1))

    def forward(self,
                m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor,
                binary: torch.Tensor, geometry: torch.Tensor, language: torch.Tensor,
                prior_tri: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        All inputs: (B, their_dim)
        prior_tri: (B, tri_dim) — previous triangulation state (or zeros)
        Returns: tri_emb (B, tri_dim), value_fn (B, 1)
        """
        B = m1.shape[0]
        device = m1.device

        if prior_tri is None:
            prior_tri = torch.zeros(B, self.tri_dim, device=device)

        streams = [m1, m2, m3, binary, geometry, language, prior_tri]

        # Project all 7 streams to tri_dim → (B, 7, tri_dim)
        tokens = torch.stack(
            [proj(s) for proj, s in zip(self.stream_projs, streams)],
            dim=1
        )

        # Self-attention across 7 tokens
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)

        # Aggregate: mean pool → (B, tri_dim)
        tri_emb = tokens.mean(dim=1)
        tri_emb = self.ln2(self.out_proj(tri_emb))

        # Invariant preservation: normalize to unit sphere scaled by learned param
        tri_emb = F.normalize(tri_emb, dim=-1) * self._invariant_scale.abs()

        # Value function
        value_fn = self.value_mlp(tri_emb)  # (B, 1)

        return tri_emb, value_fn
