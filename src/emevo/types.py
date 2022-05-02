from typing import Any, Dict, Union

import numpy as np

try:
    import jax.numpy as jnp

    Array = Union[np.ndarray, jnp.ndarray]
except ImportError as _:
    Array = np.ndarray

Action = Array
Info = Dict[str, Any]
Location = Array
# Only array and dict observations are supported
Observation = Union[Array, Dict[str, Array]]
