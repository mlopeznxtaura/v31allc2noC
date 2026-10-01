"""
Example dataset module for V31 training.

Copy to my_dataset.py, implement load_your_rows(), and run:
  cd v31 && python train.py --dataset-module ../my_dataset.py
"""

from dataset import V31DatasetBase
from embeddings import embed_text_simple
from transitions import Transition


class ExampleDataset(V31DatasetBase):
  def __init__(self):
    self.rows = load_your_rows()

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, idx: int) -> Transition:
    row = self.rows[idx]
    return Transition(
      hash_str=row["hash_str"],
      mission_id=int(row.get("mission_id", 1)),
      is_logical=bool(row.get("is_logical", True)),
      uptime_sec=float(row.get("uptime_sec", 0)),
      step_idx=int(row.get("step_idx", 0)),
      is_sandbox=bool(row.get("is_sandbox", True)),
      telemetry_stable=float(row.get("telemetry_stable", 1.0)),
      system_id=str(row.get("system_id", "train")),
      text_emb=embed_text_simple(row["text"], dim=128),
      action_token_id=int(row["action_token_id"]),
      reward=float(row["reward"]),
      done=bool(row.get("done", False)),
      value_target=row.get("value_target"),
      return_target=row.get("return_target"),
      next_frame_target=row.get("next_frame_target"),
      advantage_target=row.get("advantage_target"),
    )


def load_your_rows():
  """
  Return a list of dicts. Required keys per row:
    hash_str, text, action_token_id, reward
  Optional: mission_id, is_logical, uptime_sec, step_idx, is_sandbox,
            telemetry_stable, system_id, done, value_target, return_target,
            next_frame_target, advantage_target
  """
  raise NotImplementedError("Replace with your data loading logic")


def get_dataset():
  return ExampleDataset()
