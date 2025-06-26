from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer


@jax.jit
def compute_dist(
    xy: jax.Array,
    is_active: jax.Array,
    obstacle_xy: jax.Array,
) -> jax.Array:

    def dist(xy1: jax.Array, is_active: jax.Array, xy2: jax.Array) -> jax.Array:
        diff = jnp.expand_dims(xy1, axis=1) - jnp.expand_dims(xy2, axis=0)  # (N, M, 2)
        square = jnp.square(diff)
        dist = jnp.sqrt(jnp.sum(square, axis=2))
        return jnp.sum(jnp.mean(dist, axis=1) * is_active) / jnp.sum(is_active)

    return jax.vmap(dist, in_axes=(0, 0, None))(xy, is_active, obstacle_xy)


def main(logd: Path, n_states: int = 10, n_blocks: int = 100) -> None:
    npzfile = np.load(logd / "obstacles.npz")
    obstacle_xy = np.array(npzfile["obstacle_axy"])[:, 1:]
    dist_list = []
    for i in range(n_states):
        npzfile = np.load(logd / f"state-{i + 1}.npz")
        xy = npzfile["circle_axy"].astype(np.float32)[:, :, 1:]
        is_active = npzfile["circle_is_active"].astype(bool)
        block_size = xy.shape[0] // n_blocks
        for j in range(n_blocks):
            if j == n_blocks - 1:
                xyj = xy[block_size * j :]
                isaj = is_active[block_size * j :]
            else:
                xyj = xy[block_size * j : block_size * (j + 1)]
                isaj = is_active[block_size * j : block_size * (j + 1)]
            dist = compute_dist(xyj, isaj, obstacle_xy)
            dist_list.append(np.array(dist))
    dist = np.concatenate(dist_list)
    np.savez(logd / "obs-dist.npz", dist=dist)


if __name__ == "__main__":
    typer.run(main)
