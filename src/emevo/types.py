import typing as t

import numpy as np

Action = t.Union[int, np.ndarray]
Gene = np.ndarray
Observation = t.Union[np.ndarray, t.Dict[str, np.ndarray]]
# Importantly, reward is defiend as a dict of real values
Rewards = t.Dict[str, float]
