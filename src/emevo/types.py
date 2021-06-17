import typing as t

import numpy as np

Action = t.Union[int, np.ndarray]
Gene = np.ndarray
Info = t.Dict[str, float]
Observation = t.Union[np.ndarray, t.Dict[str, np.ndarray]]
