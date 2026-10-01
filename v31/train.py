"""
V31 from-scratch training loop.

Usage:
  cd v31
  python train.py --smoke-test                    # verify pipeline (synthetic data)
  python train.py --dataset-module ../my_data.py  # your dataset
  python train.py --dataset-module ../my_data.py --resume checkpoints/checkpoint_latest.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from checkpoint import load_training_checkpoint, save_checkpoint, write_training_manifest
from dataset import load_dataset_from_module, SyntheticSmokeDataset, V31DatasetBase
from ewc import EWCPenalty
from replay import ReplayBuffer
from transitions import Transition
from v31 import V31

ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def forward_transition(model: V31, transition: Transition, device: torch.device) -> Dict[str, Any]:
    text_emb = transition.text_emb_2d().to(device)
    return model.forward(
        hash_str=transition.hash_str,
        mission_id=transition.mission_id,
        is_logical=transition.is_logical,
        uptime_sec=transition.uptime_sec,
        step_idx=transition.step_idx,
        is_sandbox=transition.is_sandbox,
        telemetry_stable=transition.telemetry_stable,
        system_id=transition.system_id,
        text_emb=text_emb,
        temperature=transition.temperature,
        batch_size=1,
    )


def forward_batch(model: V31, transitions: List[Transition], device: torch.device) -> Dict[str, torch.Tensor]:
    outs = [forward_transition(model, t, device) for t in transitions]
    return {
        "language_token_logits": torch.cat([o["language_token_logits"] for o in outs], dim=0),
        "next_frame_prediction": torch.cat([o["next_frame_prediction"] for o in outs], dim=0),
        "value_fn": torch.cat([o["value_fn"] for o in outs], dim=0),
        "q_value": torch.cat([o["q_value"] for o in outs], dim=0),
        "advantage": torch.cat([o["advantage"] for o in outs], dim=0),
        "tri_emb": torch.cat([o["tri_emb"] for o in outs], dim=0),
    }


def compute_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    next_frame_weight: float = 0.5,
    dueling_weight: float = 0.05,
) -> Dict[str, torch.Tensor]:
    logits = out["language_token_logits"]
    actions = batch["action_token_id"].to(logits.device)
    policy_loss = F.cross_entropy(logits, actions)

    value_fn = out["value_fn"].squeeze(-1)
    if batch["return_target"] is not None:
        value_loss = F.mse_loss(value_fn, batch["return_target"].to(value_fn.device))
    elif batch["value_target"] is not None:
        value_loss = F.mse_loss(value_fn, batch["value_target"].to(value_fn.device))
    else:
        value_loss = F.mse_loss(value_fn, batch["reward"].to(value_fn.device))

    if batch["next_frame_target"] is not None:
        target = batch["next_frame_target"].to(out["next_frame_prediction"].device)
        if target.shape != out["next_frame_prediction"].shape:
            if target.shape[-1] == out["next_frame_prediction"].shape[-1]:
                next_frame_loss = F.mse_loss(out["next_frame_prediction"], target)
            else:
                next_frame_loss = torch.tensor(0.0, device=value_fn.device)
        else:
            next_frame_loss = F.mse_loss(out["next_frame_prediction"], target)
    else:
        next_frame_loss = torch.tensor(0.0, device=value_fn.device)

    q = out["q_value"].squeeze(-1)
    v = out["value_fn"].squeeze(-1)
    adv = out["advantage"].squeeze(-1)
    if batch["advantage_target"] is not None:
        dueling_loss = F.mse_loss(adv, batch["advantage_target"].to(adv.device))
    else:
        dueling_loss = F.mse_loss(q, v + adv)

    total = (
        policy_weight * policy_loss
        + value_weight * value_loss
        + next_frame_weight * next_frame_loss
        + dueling_weight * dueling_loss
    )
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "next_frame": next_frame_loss,
        "dueling": dueling_loss,
    }


def fill_replay_from_dataset(
    dataset: V31DatasetBase,
    replay: ReplayBuffer,
    model: V31,
    device: torch.device,
    reset_per_episode: bool = True,
) -> int:
    """Load all transitions into replay; optionally reset model each episode."""
    added = 0
    if reset_per_episode:
        for episode in dataset.episodes():
            model.reset_episode()
            for t in episode:
                replay.push(t)
                added += 1
    else:
        for i in range(len(dataset)):
            replay.push(dataset[i])
            added += 1
    return added


def train(args: argparse.Namespace) -> None:
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(ROOT / config_path)
    config = load_config(config_path)

    device = torch.device(args.device or config.get("device", "cpu"))
    batch_size = args.batch_size or config.get("batch_size", 16)
    lr = args.lr or config.get("learning_rate", 3e-4)
    grad_clip = args.grad_clip or config.get("grad_clip", 1.0)
    replay_capacity = config.get("replay_capacity", 512)
    ewc_lambda = config.get("ewc_lambda", 400.0)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = V31(config).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=replay_capacity)
    ewc = EWCPenalty(model, ewc_lambda=ewc_lambda)

    global_step = 0
    start_epoch = 0

    resume_path = args.resume
    if resume_path:
        step, epoch, _ = load_training_checkpoint(resume_path, model, optimizer, map_location=str(device))
        global_step = step
        start_epoch = epoch
        print(f"[train] Resumed from {resume_path} (step={step}, epoch={epoch})")

    if args.smoke_test:
        ot = config.get("output_tokens", {})
        dataset: V31DatasetBase = SyntheticSmokeDataset(
            n=args.smoke_samples,
            next_frame_dim=int(ot.get("next_frame_dim", 256)),
        )
        print(f"[train] Smoke test mode — {len(dataset)} synthetic transitions")
    elif args.dataset_module:
        dataset = load_dataset_from_module(args.dataset_module)
        print(f"[train] Loaded dataset from {args.dataset_module} ({len(dataset)} transitions)")
    else:
        print(
            "[train] No dataset provided.\n"
            "  Implement V31DatasetBase and run:\n"
            "    python train.py --dataset-module path/to/your_dataset.py\n"
            "  Or verify the pipeline:\n"
            "    python train.py --smoke-test\n"
        )
        sys.exit(1)

    write_training_manifest(str(checkpoint_dir), config)

    for epoch in range(start_epoch, args.epochs):
        replay.clear()
        n_added = fill_replay_from_dataset(dataset, replay, model, device, reset_per_episode=True)
        if n_added < batch_size:
            print(f"[train] Warning: only {n_added} transitions — need at least {batch_size} for one batch")

        epoch_loss = 0.0
        steps_this_epoch = 0
        t0 = time.time()

        for _ in range(args.steps_per_epoch):
            batch = replay.sample(batch_size)
            if batch is None:
                break

            optimizer.zero_grad(set_to_none=True)
            out = forward_batch(model, batch["transitions"], device)
            losses = compute_losses(
                out,
                batch,
                policy_weight=args.policy_weight,
                value_weight=args.value_weight,
                next_frame_weight=args.next_frame_weight,
            )
            loss = losses["total"]

            if args.ewc and ewc.is_active():
                loss = loss + ewc.penalty()

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            steps_this_epoch += 1
            epoch_loss += float(loss.item())

            if args.ewc_fisher_steps > 0 and global_step <= args.ewc_fisher_steps:
                ewc.accumulate_fisher(losses["total"])

            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} policy={losses['policy'].item():.4f} "
                    f"value={losses['value'].item():.4f} nf={losses['next_frame'].item():.4f}"
                )

            if global_step % args.save_every == 0:
                ckpt = checkpoint_dir / "checkpoint_latest.pth"
                save_checkpoint(
                    str(ckpt),
                    model,
                    optimizer,
                    step=global_step,
                    epoch=epoch,
                    config=config,
                    extra={"loss": float(loss.item())},
                )

        if args.ewc_fisher_steps > 0 and global_step >= args.ewc_fisher_steps and not ewc.is_active():
            ewc.normalize_fisher(args.ewc_fisher_steps)
            ewc.anchor()
            print(f"[train] EWC anchor set at step {global_step}")

        elapsed = time.time() - t0
        mean_loss = epoch_loss / max(steps_this_epoch, 1)
        print(
            f"[train] Epoch {epoch} done — steps={steps_this_epoch} mean_loss={mean_loss:.4f} "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(
            str(checkpoint_dir / "checkpoint_latest.pth"),
            model,
            optimizer,
            step=global_step,
            epoch=epoch + 1,
            config=config,
            extra={"epoch_mean_loss": mean_loss},
        )
        save_checkpoint(
            str(checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"),
            model,
            optimizer,
            step=global_step,
            epoch=epoch + 1,
            config=config,
        )

    weights_out = checkpoint_dir / "v31.pt"
    torch.save(model.state_dict(), weights_out)
    print(f"[train] Finished. Weights -> {weights_out}")
    print(f"[train] Full checkpoint -> {checkpoint_dir / 'checkpoint_latest.pth'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train V31 from scratch")
    p.add_argument("--config", default="config/default.json")
    p.add_argument("--dataset-module", default=None, help="Python file with get_dataset()")
    p.add_argument("--smoke-test", action="store_true", help="Run on synthetic data")
    p.add_argument("--smoke-samples", type=int, default=64)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="checkpoint_latest.pth path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--next-frame-weight", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--ewc", action="store_true", help="Apply EWC after fisher accumulation")
    p.add_argument("--ewc-fisher-steps", type=int, default=0, help="Steps to estimate Fisher before anchor")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
