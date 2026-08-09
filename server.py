"""
V31 — production API + static UI for app2.nextaura.us
"""
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
V31_DIR = ROOT / "v31"
sys.path.insert(0, str(V31_DIR))

from v31 import load_v31  # noqa: E402

STATIC_DIR = ROOT / "static"
CONFIG_PATH = V31_DIR / "config" / "default.json"

app = FastAPI(title="V31 Architecture", version="31.0.0")
_model: Optional[Any] = None


def get_model():
    global _model
    if _model is None:
        _model = load_v31(str(CONFIG_PATH), None)
    return _model


def embed_text_simple(text: str, dim: int = 128) -> torch.Tensor:
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


ARCHITECTURE = {
    "name": "V31",
    "version": "31.0",
    "summary": "7-input triangulation → 2 output tokens (NextFrame + LanguageToken).",
    "layers": [
        {
            "id": "model_scalars",
            "title": "Layer 1 — Model Scalars",
            "nodes": [
                {"id": "m1", "label": "M1 Unbounded Thinker", "desc": "Explores branches, no valence gate"},
                {"id": "m2", "label": "M2 Efficiency Zealot", "desc": "Observes M1, enforces step limits"},
                {"id": "m3", "label": "M3 Meta Observer", "desc": "Reparameterizes M1/M2, emits multipliers"},
            ],
        },
        {
            "id": "input_scalars",
            "title": "Layer 2 — Input Scalars",
            "nodes": [
                {"id": "binary", "label": "Binary", "desc": "Hash, mission, logic check"},
                {"id": "geometry", "label": "Geometry", "desc": "Sandbox/live, telemetry, system id"},
                {"id": "language", "label": "Language", "desc": "Narrative over binary + geometry"},
                {"id": "triangulation", "label": "Triangulation", "desc": "7-way cross-attention → V(s)"},
            ],
        },
        {
            "id": "outputs",
            "title": "Output Tokens",
            "nodes": [
                {"id": "next_frame", "label": "NextFramePrediction", "desc": "Predicted next state embedding"},
                {"id": "lang_token", "label": "LanguageTokenOutput", "desc": "π(a|s), Q(s,a), advantage"},
            ],
        },
    ],
}


class InferBody(BaseModel):
    text: str = Field(default="system initialized mission active", min_length=1, max_length=2000)
    sandbox: bool = True
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    mission_id: int = Field(default=1, ge=0, le=255)
    is_logical: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": "v31",
        "service": "nextaura-app2",
        "ui": "static",
    }


@app.get("/api/info")
def api_info():
    return {
        "name": "V31 Architecture",
        "repo": "https://github.com/mlopeznxtaura/v31allc2noC",
        "modes": ["infer", "architecture"],
        "architecture": ARCHITECTURE,
    }


@app.post("/api/infer")
def api_infer(body: InferBody):
    model = get_model()
    model.eval()
    text_emb = embed_text_simple(body.text, dim=128)
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.forward(
                hash_str=str(hash(body.text)),
                mission_id=body.mission_id,
                is_logical=body.is_logical,
                uptime_sec=time.time() % 86400,
                step_idx=out_step(model),
                is_sandbox=body.sandbox,
                telemetry_stable=1.0,
                system_id="app2-nextaura-us",
                text_emb=text_emb,
                temperature=body.temperature,
                batch_size=1,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (time.time() - t0) * 1000
    logits = out["language_token_logits"]
    topk = logits.topk(5)
    return {
        "ok": True,
        "latency_ms": round(elapsed_ms, 2),
        "inputs": {
            "text": body.text,
            "sandbox": body.sandbox,
            "temperature": body.temperature,
        },
        "outputs": {
            "next_frame_norm": float(out["next_frame_prediction"].norm().item()),
            "top_token_ids": topk.indices.tolist()[0],
            "top_token_logits": [round(x, 4) for x in topk.values.tolist()[0]],
        },
        "rl": {
            "value_fn": round(out["value_fn"].item(), 4),
            "q_value": round(out["q_value"].item(), 4),
            "advantage": round(out["advantage"].item(), 4),
        },
        "m3_multipliers": {
            "efficiency": round(out["efficiency_mult"].item(), 4),
            "logic": round(out["logic_mult"].item(), 4),
            "creative": round(out["creative_mult"].item(), 4),
            "efficiency_gate": round(out["efficiency_gate"].item(), 4),
        },
        "step": out["step"],
    }


def out_step(model) -> int:
    return getattr(model, "_step", 0)


@app.post("/api/reset")
def api_reset():
    model = get_model()
    model.reset_episode()
    return {"ok": True, "message": "Episode reset (M2 steps + triangulation prior cleared)"}


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
