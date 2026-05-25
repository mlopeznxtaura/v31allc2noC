"""
v31/v31.py — V31 Architecture

Full forward pass:

  LAYER 1 — Model Scalars (Meta-thinkers)
    M1: Unbounded thinker
    M2: Efficiency zealot (observes M1)
    M3: Meta-observer (observes M1 + M2, reparameterizes both)

  LAYER 2 — Input Scalars
    Binary:      machine init (hash, mission, logic check)
    Geometry:    space/time self-location (sandbox/live, telemetry, system id)
    Language:    narrative contextualization of Binary + Geometry
    Triangulation: RL value function across all 6 → 7 (with self as prior)

  OUTPUTS (2 tokens only)
    NextFramePrediction:  what happens next
    LanguageTokenOutput:  the actual RL decision/action
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from agents import M1_UnboundedThinker, M2_EfficiencyZealot, M3_MetaObserver
from core.scalars import BinaryScalar, GeometryScalar, LanguageScalar, TriangulationScalar
from core.outputs import NextFramePredictor, LanguageTokenOutput


class V31(nn.Module):
    """
    V31: 3 model scalars + 4 input scalars → Triangulation → 2 output tokens.

    Architecture invariants:
      - Triangulation receives exactly 7 inputs (6 + self-prior)
      - Only 2 token outputs emitted per forward pass
      - M3 may reparameterize M1/M2 every repar_interval steps
      - Value function output ∈ [0, 1] (sigmoid bounded)
    """

    def __init__(self, config: dict):
        super().__init__()
        c = config
        sd = c["scalar_dims"]
        ot = c["output_tokens"]
        ab = c["agent_time_budgets"]

        # ── Layer 1: Model Scalars ─────────────────────────────────────────
        self.m1 = M1_UnboundedThinker(
            d_in=sd["binary"] // 2,       # M1 sees condensed binary signal
            d_out=sd["m1"],
            max_branches=ab["m1_max_branches"]
        )
        self.m2 = M2_EfficiencyZealot(
            d_in=sd["m1"],
            d_out=sd["m2"],
            max_steps=ab["m2_max_steps"]
        )
        self.m3 = M3_MetaObserver(
            d_m1=sd["m1"],
            d_m2=sd["m2"],
            d_out=sd["m3"],
            repar_interval=ab["m3_repar_interval"]
        )

        # M1 needs an input projection from raw input
        self.m1_input_proj = nn.Linear(sd["binary"], sd["binary"] // 2)

        # ── Layer 2: Input Scalars ─────────────────────────────────────────
        self.binary_scalar = BinaryScalar(dim=sd["binary"])
        self.geo_scalar = GeometryScalar(dim=sd["geometry"])
        self.lang_scalar = LanguageScalar(
            embed_dim=sd["binary"],        # text_emb same dim as binary for simplicity
            binary_dim=sd["binary"],
            geo_dim=sd["geometry"],
            lang_dim=sd["language"]
        )
        self.triangulation = TriangulationScalar(
            m1_dim=sd["m1"],
            m2_dim=sd["m2"],
            m3_dim=sd["m3"],
            binary_dim=sd["binary"],
            geo_dim=sd["geometry"],
            lang_dim=sd["language"],
            tri_dim=sd["triangulation"],
            n_heads=c["triangulation_heads"]
        )

        # ── Output Tokens ──────────────────────────────────────────────────
        self.next_frame = NextFramePredictor(
            tri_dim=sd["triangulation"],
            out_dim=ot["next_frame_dim"]
        )
        self.lang_token = LanguageTokenOutput(
            tri_dim=sd["triangulation"],
            next_frame_dim=ot["next_frame_dim"],
            vocab_size=c["vocab_size"],
            action_dim=ot["language_token_dim"]
        )

        # Prior triangulation state (persistent across steps)
        self.register_buffer("_prior_tri",
                              torch.zeros(1, sd["triangulation"]))

        self._step = 0

    def forward(self,
                # Binary inputs
                hash_str: str,
                mission_id: int,
                is_logical: bool,
                # Geometry inputs
                uptime_sec: float,
                step_idx: int,
                is_sandbox: bool,
                telemetry_stable: float,
                system_id: str,
                # Language input (pre-embedded text vector)
                text_emb: torch.Tensor,  # (B, binary_dim) — caller provides embedding
                # Optional
                temperature: float = 1.0,
                batch_size: int = 1
                ) -> Dict[str, Any]:
        """
        Full V31 forward pass.

        Returns dict with:
          next_frame_prediction  (B, next_frame_dim)
          language_token_logits  (B, vocab_size)
          action_embedding       (B, action_dim)
          value_fn               (B, 1)
          q_value                (B, 1)
          advantage              (B, 1)
          tri_emb                (B, tri_dim)    — for inspection
          efficiency_mult        (B, 1)
          logic_mult             (B, 1)
          creative_mult          (B, 1)
        """
        self._step += 1
        device = text_emb.device

        # ── Scalar 4: Binary ──────────────────────────────────────────────
        bin_emb = self.binary_scalar(hash_str, mission_id, is_logical, batch_size)
        bin_emb = bin_emb.to(device)

        # ── Scalar 5: Geometry ────────────────────────────────────────────
        geo_emb = self.geo_scalar(uptime_sec, step_idx, is_sandbox,
                                  telemetry_stable, system_id, batch_size)
        geo_emb = geo_emb.to(device)

        # ── Scalar 6: Language ────────────────────────────────────────────
        lang_emb = self.lang_scalar(text_emb, bin_emb, geo_emb)

        # ── Layer 1: M1 ──────────────────────────────────────────────────
        m1_in = torch.relu(self.m1_input_proj(bin_emb))
        m1_emb = self.m1(m1_in)

        # ── Layer 1: M2 (observes M1) ────────────────────────────────────
        m2_emb, efficiency_gate = self.m2(m1_in, m1_emb)

        # ── Layer 1: M3 (observes M1 + M2, reparameterizes) ──────────────
        m3_emb, efficiency_mult, logic_mult, creative_mult = self.m3(m1_emb, m2_emb)
        self.m3.reparameterize(self.m1, self.m2, creative_mult, efficiency_mult)

        # ── Scalar 7: Triangulation (7 inputs → value function) ───────────
        prior = self._prior_tri.expand(batch_size, -1).to(device)
        tri_emb, value_fn = self.triangulation(
            m1_emb, m2_emb, m3_emb,
            bin_emb, geo_emb, lang_emb,
            prior_tri=prior
        )
        # Update prior for next step
        self._prior_tri = tri_emb.detach().mean(dim=0, keepdim=True)

        # ── Output Token 1: NextFramePrediction ───────────────────────────
        next_frame_pred = self.next_frame(tri_emb, value_fn)

        # ── Output Token 2: LanguageTokenOutput ───────────────────────────
        token_logits, action_emb, advantage, q_value = self.lang_token(
            tri_emb, next_frame_pred, value_fn, temperature
        )

        return {
            # ── 2 primary output tokens ──
            "next_frame_prediction": next_frame_pred,
            "language_token_logits": token_logits,
            # ── RL signals ──
            "action_embedding": action_emb,
            "value_fn": value_fn,
            "q_value": q_value,
            "advantage": advantage,
            # ── Internal state (for telemetry / M3 reparameterization) ──
            "tri_emb": tri_emb,
            "efficiency_mult": efficiency_mult,
            "logic_mult": logic_mult,
            "creative_mult": creative_mult,
            "efficiency_gate": efficiency_gate,
            "step": self._step,
        }

    def reset_episode(self):
        """Reset M2 step counter and prior tri state for new episode."""
        self.m2.reset_step()
        self._prior_tri.zero_()
        self._step = 0


def load_v31(config_path: str = "config/default.json",
             checkpoint_path: Optional[str] = None) -> "V31":
    with open(config_path) as f:
        config = json.load(f)
    model = V31(config)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"[v31] Loaded checkpoint: {checkpoint_path}")
    model.eval()
    return model
