"""Policy visualization
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Circle
from numpy.typing import NDArray
from phyjax2d import Vec2d


def draw_cf_policy(
    policy_means: NDArray,
    rotation: float,
    fig_unit: float,
    max_force: float,
) -> None:
    n_policies = policy_means.shape[0]
    if n_policies == 1:
        fig, ax = plt.subplots(figsize=(fig_unit, fig_unit))
        axes = [ax]
    elif n_policies <= 6:
        ncols = n_policies // 2
        fig, axes = plt.subplots(
            nrows=2,
            ncols=ncols,
            figsize=(ncols * fig_unit, 2 * fig_unit),
        )
    else:
        ncols = n_policies // 3
        fig, axes = plt.subplots(
            nrows=3,
            ncols=ncols,
            figsize=(ncols * fig_unit, 2 * fig_unit),
        )
    fig.tight_layout()
    # Arrow points
    center = Vec2d(max_force * 1.5, max_force * 1.5)
    unit = Vec2d(0.0, 1.0)
    u_left = unit.rotated(math.pi * 1.25 + rotation)
    u_right = unit.rotated(math.pi * 0.75 + rotation)
    s_left = u_left * max_force * 0.5 + center
    s_right = u_right * max_force * 0.5 + center
    # Draw the arrows
    for policy_mean, ax in zip(policy_means, np.ravel(axes)):
        ax.set_xlim(0, max_force * 3)
        ax.set_ylim(0, max_force * 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        # Circle
        circle = Circle((center.x, center.y), max_force * 0.5, fill=False)
        ax.add_patch(circle)
        # Left
        d_left = u_left * policy_mean[0].item()
        arrow = Arrow(
            s_left.x,
            s_left.y,
            d_left.x,
            d_left.y,
            width=max_force * 0.1,
            color="r",
        )
        ax.add_patch(arrow)
        # Right
        d_right = u_right * policy_mean[1].item()
        arrow = Arrow(
            s_right.x,
            s_right.y,
            d_right.x,
            d_right.y,
            width=max_force * 0.1,
            color="r",
        )
        ax.add_patch(arrow)

    plt.show()
