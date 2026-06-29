from __future__ import annotations

import numpy as np


def preprocess_observation(observation: np.ndarray) -> np.ndarray:
    """Convert the 96x96 RGB frame into normalized channel-first pixels."""
    frame = np.asarray(observation, dtype=np.float32) / 255.0
    return np.transpose(frame, (2, 0, 1))
