# V31 Architecture — Schema Reference

## Pivot from V21 → V31

V21 had: Binary + Geometry + Language → CrossModalAttention → ScalarEngine
V31 has: 7-input triangulation → 2 output tokens only. No extra heads.

---

## Full Data Flow

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         V31 FORWARD PASS                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    LAYER 1: MODEL SCALARS                        │    ║
║  │                                                                  │    ║
║  │  [RAW_INPUT] ──→ M1: UnboundedThinker ──────────────────────┐  │    ║
║  │                    (explores all branches, no valence gate)    │  │    ║
║  │                         │                                      │  │    ║
║  │                         ▼                                      │  │    ║
║  │                  M2: EfficiencyZealot  ──────────────────────┐│  │    ║
║  │                    (observes M1, enforces time rules)         ││  │    ║
║  │                         │                                     ││  │    ║
║  │                         ▼                                     ▼▼  │    ║
║  │                  M3: MetaObserver ─────────────────────────────┘  │    ║
║  │                    (observes M1+M2, manages time budget,          │    ║
║  │                     emits: efficiency×, logic×, creative×)        │    ║
║  │                         │ reparameterize ↓                        │    ║
║  │                    M1.branch_budget ← creative×                   │    ║
║  │                    M2.max_steps    ← efficiency×                  │    ║
║  └──────────────────────────┬────────────────────────────────────────┘    ║
║                             │ M1_emb, M2_emb, M3_emb                      ║
║                             ▼                                             ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    LAYER 2: INPUT SCALARS                        │    ║
║  │                                                                  │    ║
║  │  Scalar 4: Binary    ← hash_str, mission_id, is_logical         │    ║
║  │  Scalar 5: Geometry  ← uptime, step, sandbox, telemetry, sysid  │    ║
║  │  Scalar 6: Language  ← text_emb + Binary + Geometry             │    ║
║  │                                                                  │    ║
║  │  Scalar 7: TRIANGULATION                                        │    ║
║  │  ┌──────────────────────────────────────────────────────────┐  │    ║
║  │  │  Receives ALL 7: M1 + M2 + M3 + Binary + Geo + Lang +   │  │    ║
║  │  │                  self_prior (prior triangulation state)   │  │    ║
║  │  │                                                            │  │    ║
║  │  │  → CrossAttention(7 tokens) → mean_pool → normalize      │  │    ║
║  │  │  → tri_emb (B, 256) [unit sphere × learned scale]         │  │    ║
║  │  │  → value_fn (B, 1) ∈ [0,1]  ← RL state value V(s)       │  │    ║
║  │  └──────────────────────────────────────────────────────────┘  │    ║
║  └──────────────────────────┬────────────────────────────────────────┘    ║
║                             │ tri_emb, value_fn                           ║
║                             ▼                                             ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    OUTPUT TOKENS (×2 only)                       │    ║
║  │                                                                  │    ║
║  │  Token 1: NextFramePrediction                                   │    ║
║  │    f(tri_emb, value_fn) → next_state_emb (B, 256)              │    ║
║  │    residual: next ≈ current + 0.1 * delta                       │    ║
║  │                                                                  │    ║
║  │  Token 2: LanguageTokenOutput                                   │    ║
║  │    π(a|s) = policy(tri_emb || next_frame)                       │    ║
║  │    → token_logits  (B, vocab_size=32768)  ← discrete action     │    ║
║  │    → action_emb    (B, 256)               ← continuous action   │    ║
║  │    → advantage A(s,a) = Q(s,a) - V(s)    ← dueling head        │    ║
║  │    → q_value   Q(s,a) = V(s) + A(s,a)                          │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Invariants

| Property | Guarantee |
|---|---|
| Output count | Exactly 2 tokens per forward pass |
| Value function range | V(s) ∈ [0, 1] (sigmoid) |
| Triangulation magnitude | Unit sphere × learned scale (F.normalize) |
| M3 time budget | Self-decaying: budget *= 0.99/step, floor 0.05 |
| M1 branch count | Bounded [1, max_branches]; set by M3 every repar_interval |
| M2 step kill | Hard zero gate after max_steps exceeded |

---

## Delta from V21

| Component | V21 | V31 |
|---|---|---|
| Model agents | None | M1, M2, M3 (Layer 1) |
| Input scalars | Binary + Geo + Lang (3) | Binary + Geo + Lang + Triangulation (4) |
| Cross-attention | CrossModalAttention (3 tokens) | TriangulationScalar (7 tokens) |
| Value function | None | RL V(s) from Triangulation |
| Output heads | Decoder + GGUF + vLLM export | 2 tokens only: NextFrame + LangToken |
| RL signals | None | V(s), Q(s,a), A(s,a) per step |
| Reparameterization | None | M3 → M1.branches, M2.steps |

---

## Files

```
v31/
├── config/default.json          ← scalar dims, time budgets, output dims
├── agents.py                    ← M1, M2, M3
├── core/
│   ├── scalars.py               ← Binary, Geometry, Language, Triangulation
│   └── outputs.py               ← NextFramePredictor, LanguageTokenOutput
├── v31.py                       ← V31(nn.Module) — full wiring + load_v31()
├── infer.py                     ← CLI inference runner
└── ARCHITECTURE.md              ← this file
```
