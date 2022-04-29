import typing as t

import numpy as np

Action = t.Union[int, np.ndarray]
Info = t.Dict[str, t.Union[int, float]]
Observation = t.Union[np.ndarray, t.Dict[str, np.ndarray]]
RGB = t.Tuple[int, int, int]
