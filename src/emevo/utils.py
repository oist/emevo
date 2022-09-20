from __future__ import annotations

import numpy as np
from numpy.random import Generator


def sample_location(
    gen: Generator,
    center: tuple[float, float],
    radius_max: float,
    radius_min: float = 0.0,
) -> tuple[float, float]:
    cx, cy = center
    theta = gen.random() * 2.0 * np.pi
    radius_range = radius_max - radius_min
    radius = np.sqrt(gen.random()) * radius_range + radius_min
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    return x, y
