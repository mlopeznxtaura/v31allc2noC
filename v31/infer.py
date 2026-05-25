"""
v31/infer.py — Run a single V31 forward pass

Usage:
    python infer.py
    python infer.py --text "system is live, mission active" --sandbox false
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import json
import time
import argparse
from v31 import load_v31


def embed_text_simple(text: str, dim: int = 128) -> torch.Tensor:
    """Minimal text → vector without external deps. Hash-based embedding."""
    import hashlib
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
    return torch.tensor([flat], dtype=torch.float32)  # (1, dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="system initialized mission active logic verified", type=str)
    parser.add_argument("--sandbox", default="true", choices=["true", "false"])
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--temperature", default=1.0, type=float)
    args = parser.parse_args()

    print(f"\n[V31] Loading model from {args.config}")
    model = load_v31(args.config, args.checkpoint)
    model.eval()

    text_emb = embed_text_simple(args.text, dim=128)
    is_sandbox = args.sandbox == "true"

    print(f"[V31] Forward pass — text='{args.text}' sandbox={is_sandbox}")
    t0 = time.time()

    with torch.no_grad():
        out = model.forward(
            hash_str=str(hash(args.text)),
            mission_id=1,
            is_logical=True,
            uptime_sec=time.time() % 86400,
            step_idx=0,
            is_sandbox=is_sandbox,
            telemetry_stable=1.0,
            system_id="v31-default",
            text_emb=text_emb,
            temperature=args.temperature,
            batch_size=1,
        )

    elapsed = (time.time() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"  V31 OUTPUT TOKENS")
    print(f"{'='*60}")
    print(f"  NextFramePrediction shape : {out['next_frame_prediction'].shape}")
    print(f"  NextFramePrediction norm  : {out['next_frame_prediction'].norm().item():.4f}")
    print(f"  LanguageToken top-5 ids   : {out['language_token_logits'].topk(5).indices.tolist()}")
    print(f"  LanguageToken top-5 logits: {out['language_token_logits'].topk(5).values.tolist()}")
    print(f"\n  RL Signals")
    print(f"  {'─'*40}")
    print(f"  value_fn    : {out['value_fn'].item():.4f}")
    print(f"  q_value     : {out['q_value'].item():.4f}")
    print(f"  advantage   : {out['advantage'].item():.4f}")
    print(f"\n  M3 Multipliers")
    print(f"  {'─'*40}")
    print(f"  efficiency  : {out['efficiency_mult'].item():.4f}")
    print(f"  logic       : {out['logic_mult'].item():.4f}")
    print(f"  creative    : {out['creative_mult'].item():.4f}")
    print(f"  eff_gate    : {out['efficiency_gate'].item():.4f}")
    print(f"\n  Step        : {out['step']}")
    print(f"  Latency     : {elapsed:.2f}ms")
    print(f"{'='*60}\n")

    # Save to telemetry
    os.makedirs("telemetry", exist_ok=True)
    import json as _json
    record = {
        "step": out["step"],
        "value_fn": out["value_fn"].item(),
        "q_value": out["q_value"].item(),
        "advantage": out["advantage"].item(),
        "efficiency_mult": out["efficiency_mult"].item(),
        "logic_mult": out["logic_mult"].item(),
        "creative_mult": out["creative_mult"].item(),
        "top5_token_ids": out["language_token_logits"].topk(5).indices.tolist()[0],
        "latency_ms": elapsed,
        "timestamp": time.time(),
    }
    with open("telemetry/v31_infer.jsonl", "a") as f:
        f.write(_json.dumps(record) + "\n")
    print(f"[V31] Telemetry written → telemetry/v31_infer.jsonl")


if __name__ == "__main__":
    main()
