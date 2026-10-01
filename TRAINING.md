# V31 training (from scratch)

Train the full V31 architecture on your own dataset. The repo provides the loop, losses, replay, EWC, and checkpoints — you provide transitions.

## Quick verify (synthetic smoke test)

```bash
cd v31
pip install torch
python train.py --smoke-test --epochs 2 --steps-per-epoch 10
```

Outputs under `v31/checkpoints/`:
- `checkpoint_latest.pth` — full state (model + optimizer + step)
- `checkpoint_latest.pt` — weights only (for deploy)
- `v31.pt` — final weights after training run

## Your dataset

1. Copy `user_dataset.example.py` → `my_dataset.py`
2. Implement `load_your_rows()` or subclass `V31DatasetBase`
3. Each sample is a `Transition` with at least:
   - `hash_str`, `text` (embedded via `embed_text_simple`), `action_token_id`, `reward`
   - geometry fields: `uptime_sec`, `step_idx`, `is_sandbox`, etc.
   - optional targets: `value_target`, `return_target`, `next_frame_target`

```bash
cd v31
python train.py --dataset-module ../my_dataset.py --epochs 50 --steps-per-epoch 200
```

## Resume training

```bash
python train.py --dataset-module ../my_dataset.py --resume checkpoints/checkpoint_latest.pth
```

## Continual learning (EWC)

After an initial training phase, anchor weights and penalize drift:

```bash
python train.py --dataset-module ../phase1.py --ewc --ewc-fisher-steps 200
# later, on new mission data:
python train.py --dataset-module ../phase2.py --resume checkpoints/checkpoint_latest.pth --ewc
```

## Inference with trained weights

```bash
cd v31
python infer.py --checkpoint checkpoints/v31.pt --text "mission active"
```

## Deploy app2 with checkpoint

Set env before starting the server / container:

```bash
export V31_CHECKPOINT=/path/to/checkpoints/v31.pt
python server.py
```

Or mount `checkpoints/v31.pt` in Docker and set `V31_CHECKPOINT`.

## Losses

| Head | Loss |
|------|------|
| `language_token_logits` | cross-entropy vs `action_token_id` |
| `value_fn` | MSE vs `return_target`, `value_target`, or `reward` |
| `next_frame_prediction` | MSE vs `next_frame_target` (when provided) |
| dueling heads | MSE aligning `q_value` ≈ `value_fn + advantage` |

Weights: `--policy-weight`, `--value-weight`, `--next-frame-weight`
