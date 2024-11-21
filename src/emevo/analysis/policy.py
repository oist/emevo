"""Policy visualization"""

import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Circle
from numpy.typing import NDArray
from phyjax2d import Vec2d


def draw_cf_policy(
    names: list[str],
    policy_means: NDArray,
    rotation: float,
    fig_unit: float,
    max_force: float,
    show: bool = True,
) -> None:
    n_policies = len(names)
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
    d_unit = unit.rotated(rotation)
    s_left = unit.rotated(math.pi * 1.25 + rotation) * max_force * 0.5 + center
    s_right = unit.rotated(math.pi * 0.75 + rotation) * max_force * 0.5 + center
    # Draw the arrows
    for title, policy_mean, ax in zip(names, policy_means, np.ravel(axes)):
        # Misc
        ax.set_xlim(0, max_force * 3)
        ax.set_ylim(0, max_force * 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        # Circle
        circle = Circle((center.x, center.y), max_force * 0.5, fill=False)
        ax.add_patch(circle)
        # Left
        d_left = d_unit * policy_mean[0].item()
        s_left_shifted = s_left - d_left
        arrow = Arrow(
            s_left_shifted.x,
            s_left_shifted.y,
            d_left.x,
            d_left.y,
            # 10% of the width? Looks thinner...
            width=max_force * 0.3,
            color="r",
        )
        ax.add_patch(arrow)
        # Right
        d_right = d_unit * policy_mean[1].item()
        s_right_shifted = s_right - d_right
        arrow = Arrow(
            s_right_shifted.x,
            s_right_shifted.y,
            d_right.x,
            d_right.y,
            width=max_force * 0.3,
            color="r",
        )
        ax.add_patch(arrow)

    if show:
        plt.show()


def draw_cf_policy_multi(
    names: list[str],
    rotations: list[float],
    policy_means: NDArray,  # (N-obs, N-agents, LR)
    fig_unit: float,
    max_force: float,
    show: bool = True,
) -> None:
    n_policies = len(names)
    n_obs, n_policies = policy_means.shape[:2]
    observations = [f"Observation {i+1}" for i in range(n_obs)]
    fig, axes = plt.subplots(
        nrows=n_policies,
        ncols=n_obs,
        figsize=(n_obs * fig_unit, n_policies * fig_unit),
    )
    fig.tight_layout()
    # Arrow points
    center = Vec2d(max_force * 1.5, max_force * 1.5)
    unit = Vec2d(0.0, 1.0)
    # Draw the arrows
    for j, (obs_title, rot) in enumerate(zip(observations, rotations)):
        d_unit = unit.rotated(rot)
        s_left = unit.rotated(math.pi * 1.25 + rot) * max_force * 0.5 + center
        s_right = unit.rotated(math.pi * 0.75 + rot) * max_force * 0.5 + center
        for i, policy_mean in enumerate(policy_means[j]):
            ax = axes[i][j]
            ax.set_xlim(0, max_force * 3)
            ax.set_ylim(0, max_force * 3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            if i == 0:
                ax.set_title(obs_title)
            if j == 0:
                ax.set_ylabel(names[i])
            # Circle
            circle = Circle((center.x, center.y), max_force * 0.5, fill=False)
            ax.add_patch(circle)
            # Left
            d_left = d_unit * policy_mean[0].item()
            s_left_shifted = s_left - d_left
            arrow = Arrow(
                s_left_shifted.x,
                s_left_shifted.y,
                d_left.x,
                d_left.y,
                # 10% of the width? Looks thinner...
                width=max_force * 0.3,
                color="r",
            )
            ax.add_patch(arrow)
            # Right
            d_right = d_unit * policy_mean[1].item()
            s_right_shifted = s_right - d_right
            arrow = Arrow(
                s_right_shifted.x,
                s_right_shifted.y,
                d_right.x,
                d_right.y,
                width=max_force * 0.3,
                color="r",
            )
            ax.add_patch(arrow)

    if show:
        plt.show()
