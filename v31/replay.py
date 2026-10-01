"""Experience replay buffer for V31 training."""

import random
from collections import deque
from typing import Deque, Dict, List, Optional

from transitions import Transition, collate_transitions


class ReplayBuffer:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def push_many(self, transitions: List[Transition]) -> None:
        for t in transitions:
            self.push(t)

    def sample(self, batch_size: int) -> Optional[Dict]:
        if len(self._buffer) < batch_size:
            return None
        batch = random.sample(list(self._buffer), batch_size)
        return collate_transitions(batch)

    def clear(self) -> None:
        self._buffer.clear()
