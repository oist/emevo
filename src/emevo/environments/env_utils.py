"""Place agent and food"""

from __future__ import annotations

import dataclasses
import enum
import functools
from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.phyjax2d import ShapeDict, StateDict, circle_overlap

Self = Any


@chex.dataclass
class FoodNumState:
    current: jax.Array
    internal: jax.Array

    def n_max_recover(self) -> jax.Array:
        internal_int = jnp.floor(self.internal).astype(jnp.int32)
        return jnp.clip(internal_int - self.current, a_min=0)

    def eaten(self, n: int | jax.Array) -> Self:
        return FoodNumState(
            current=self.current - n,
            internal=self.internal - n,
        )

    def recover(self, n: int | jax.Array = 1) -> Self:
        return dataclasses.replace(self, current=self.current + n)

    def _update(self, internal: jax.Array) -> Self:
        return FoodNumState(
            current=self.current,
            internal=internal,
        )

    def get_slice(self, index: int) -> Self:
        return jax.tree_util.tree_map(lambda x: x[index], self)


class FoodNumFn(Protocol):
    initial: int

    def __call__(self, n_steps: int, state: FoodNumState) -> FoodNumState: ...


@dataclasses.dataclass(frozen=True)
class FoodNumConstant:
    initial: int

    def __call__(self, _: int, state: FoodNumState) -> FoodNumState:
        # Do nothing here
        return state._update(jnp.array(self.initial, dtype=jnp.float32))


@dataclasses.dataclass(frozen=True)
class FoodNumLinear:
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, _: int, state: FoodNumState) -> FoodNumState:
        # Increase the number of foods by dn_dt
        internal = jnp.fmax(state.current, state.internal)
        max_value = jnp.array(self.capacity, dtype=jnp.float32)
        return state._update(jnp.clip(internal + self.growth_rate, a_max=max_value))


@dataclasses.dataclass(frozen=True)
class FoodNumLogistic:
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, _: int, state: FoodNumState) -> FoodNumState:
        internal = jnp.fmax(state.current, state.internal)
        dn_dt = self.growth_rate * internal * (1 - internal / self.capacity)
        return state._update(internal + dn_dt)


class FoodNumCycle:
    def __init__(
        self,
        interval: int,
        *num_fns: tuple[str, ...] | FoodNumFn,
    ) -> None:
        numfn_list = []
        for fn_or_base in num_fns:
            if callable(fn_or_base):
                numfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                fn, _ = FoodNum(name)(*args)
                numfn_list.append(fn)
        self._numfn_list = numfn_list
        self._n_fn = len(numfn_list)
        self._interval = interval

    @property
    def initial(self) -> int:
        return self._numfn_list[0].initial

    def __call__(self, n_steps: int, state: FoodNumState) -> FoodNumState:
        n = n_steps // self._interval
        index = n % self._n_fn
        return jax.lax.switch(index, self._numfn_list, n_steps, state)


class FoodNumScheduled:
    """Branching based on steps."""

    def __init__(
        self,
        intervals: int | list[int],
        *num_fns: tuple[str, ...] | FoodNumFn,
    ) -> None:
        numfn_list = []
        for fn_or_base in num_fns:
            if callable(fn_or_base):
                numfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                fn, _ = FoodNum(name)(*args)
                numfn_list.append(fn)
        self._numfn_list = numfn_list
        if isinstance(intervals, int):
            intervals = [intervals * (i + 1) for i in range(len(self._numfn_list))]
        self._intervals = jnp.array(intervals, dtype=jnp.int32)

    @property
    def initial(self) -> int:
        return self._numfn_list[0].initial

    def __call__(self, n_steps: int, state: FoodNumState) -> FoodNumState:
        index = jnp.digitize(n_steps, bins=self._intervals)
        return jax.lax.switch(index, self._numfn_list, n_steps, state)


class FoodNum(str, enum.Enum):
    """Methods to determine the number of foods reproduced."""

    CONSTANT = "constant"
    CYCLE = "cycle"
    LINEAR = "linear"
    LOGISTIC = "logistic"
    SCHEDULED = "scheduled"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[FoodNumFn, FoodNumState]:
        if self is FoodNum.CONSTANT:
            fn = FoodNumConstant(*args, **kwargs)
        elif self is FoodNum.CYCLE:
            fn = FoodNumCycle(*args, **kwargs)
        elif self is FoodNum.LINEAR:
            fn = FoodNumLinear(*args, **kwargs)
        elif self is FoodNum.LOGISTIC:
            fn = FoodNumLogistic(*args, **kwargs)
        elif self is FoodNum.SCHEDULED:
            fn = FoodNumScheduled(*args, **kwargs)
        else:
            raise AssertionError("Unreachable")

        initial = fn.initial
        state = FoodNumState(
            current=jnp.array(0, dtype=jnp.int32),
            internal=jnp.array(float(initial), dtype=jnp.float32),
        )
        return cast(FoodNumFn, fn), state


class Coordinate(Protocol):
    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

    def contains_circle(
        self,
        center: jax.Array,
        radius: jax.Array | float,
    ) -> jax.Array: ...

    def uniform(self, key: chex.PRNGKey) -> jax.Array: ...


@dataclasses.dataclass
class CircleCoordinate(Coordinate):
    center: tuple[float, float]
    radius: float

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        cx, cy = self.center
        r = self.radius
        return (cx - r, cx + r), (cy - r, cy + r)

    def contains_circle(
        self,
        center: jax.Array,
        radius: jax.Array | float,
    ) -> jax.Array:
        a2b = center - jnp.array(self.center)
        distance = jnp.linalg.norm(a2b, ord=2)
        return distance + radius <= self.radius

    def uniform(self, key: chex.PRNGKey) -> jax.Array:
        low = jnp.array([0.0, 0.0])
        high = jnp.array([1.0, 2.0 * jnp.pi])
        squared_norm, angle = jax.random.uniform(
            key,
            shape=(2,),
            minval=low,
            maxval=high,
        )
        radius = self.radius * jnp.sqrt(squared_norm)
        cx, cy = self.center
        return jnp.array([radius * jnp.cos(angle) + cx, radius * jnp.sin(angle) + cy])


@dataclasses.dataclass
class SquareCoordinate(Coordinate):
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self.xlim, self.ylim

    def contains_circle(
        self, center: jax.Array, radius: jax.Array | float
    ) -> jax.Array:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = jnp.array([xmin, ymin]) + radius
        high = jnp.array([xmax, ymax]) - radius
        return jnp.logical_and(jnp.all(low <= center), jnp.all(center <= high))

    def uniform(self, key: chex.PRNGKey) -> jax.Array:
        xmin, xmax = self.xlim
        ymin, ymax = self.ylim
        low = jnp.array([xmin, ymin])
        high = jnp.array([xmax, ymax])
        return jax.random.uniform(key, shape=(2,), minval=low, maxval=high)


@chex.dataclass
class LocatingState:
    n_produced: jax.Array

    def increment(self, n: jax.Array | int = 1) -> Self:
        return LocatingState(n_produced=self.n_produced + n)

    def get_slice(self, index: int) -> Self:
        return jax.tree_util.tree_map(lambda x: x[index], self)


LocatingFn = Callable[[chex.PRNGKey, int, LocatingState], jax.Array]


class Locating(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    GAUSSIAN_LINEAR = "gaussian-linear"
    GAUSSIAN_MIXTURE = "gaussian-mixture"
    PERIODIC = "periodic"
    CHOICE = "choice"
    SCHEDULED = "scheduled"
    LINEAR = "linear"
    SWITCHING = "switching"
    UNIFORM = "uniform"
    UNIFORM_LINEAR = "uniform-linear"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[LocatingFn, LocatingState]:
        # Make sure it has shape () because it's also used for agents as singleton
        state = LocatingState(
            n_produced=jnp.array(0, dtype=jnp.int32),
        )
        if self is Locating.GAUSSIAN:
            return LocGaussian(*args, **kwargs), state
        elif self is Locating.GAUSSIAN_LINEAR:
            return LocGaussianLinear(*args, **kwargs), state
        elif self is Locating.GAUSSIAN_MIXTURE:
            return LocGaussianMixture(*args, **kwargs), state
        elif self is Locating.PERIODIC:
            return LocPeriodic(*args, **kwargs), state
        elif self is Locating.CHOICE:
            return LocChoice(*args, **kwargs), state
        elif self is Locating.UNIFORM:
            return LocUniform(*args, **kwargs), state
        elif self is Locating.UNIFORM_LINEAR:
            return LocUniformLinear(*args, **kwargs), state
        elif self is Locating.SCHEDULED:
            return LocScheduled(*args, **kwargs), state
        elif self is Locating.SWITCHING:
            return LocSwitching(*args, **kwargs), state
        else:
            raise AssertionError("Unreachable")


class LocGaussian:
    def __init__(self, mean: ArrayLike, stddev: ArrayLike) -> None:
        self.mean = jnp.array(mean)
        self.stddev = jnp.array(stddev)
        self.shape = self.mean.shape

    def __call__(
        self, key: chex.PRNGKey, _n_steps: int, _state: LocatingState
    ) -> jax.Array:
        del _n_steps, _state
        return jax.random.normal(key, shape=self.shape) * self.stddev + self.mean


class LocGaussianLinear:
    def __init__(
        self,
        mean: ArrayLike,
        stddev: ArrayLike,
        modulation: ArrayLike,
        clip: ArrayLike,
    ) -> None:
        self.mean = jnp.array(mean)
        self.stddev = jnp.array(stddev)
        self.modulation = jnp.array(modulation)
        self.clip = jnp.array(clip)
        self.shape = self.mean.shape

    def __call__(
        self,
        key: chex.PRNGKey,
        n_steps: int,
        _state: LocatingState,
    ) -> jax.Array:
        del _state
        mean = jnp.clip(self.mean + self.modulation * n_steps, a_min=0.0, a_max=self.clip)
        return jax.random.normal(key, shape=self.shape) * self.stddev + mean


class LocUniformLinear:
    def __init__(
        self,
        coordinate_or_list: Coordinate | list[float],
        modulation: ArrayLike,
        clip: ArrayLike,
    ) -> None:
        if isinstance(coordinate_or_list, list | tuple):
            self.coordinate = SquareCoordinate(
                tuple(coordinate_or_list[:2]),  # type: ignore
                tuple(coordinate_or_list[2:]),  # type: ignore
            )
        else:
            self.coordinate = coordinate_or_list
        self.modulation = jnp.array(modulation)
        self.clip = jnp.array(clip)

    def __call__(
        self,
        key: chex.PRNGKey,
        n_steps: int,
        _state: LocatingState,
    ) -> jax.Array:
        del _state
        sampled = self.coordinate.uniform(key)
        return jnp.clip(sampled + self.modulation * n_steps, min=0.0, max=self.clip)


class LocGaussianMixture:
    def __init__(
        self,
        probs: ArrayLike,
        mean_arr: ArrayLike,
        stddev_arr: ArrayLike,
    ) -> None:
        self.mean = jnp.array(mean_arr)
        self.stddev = jnp.array(stddev_arr)
        self.probs = jnp.array(probs)
        self.n = self.probs.shape[0]

    def __call__(
        self,
        key: chex.PRNGKey,
        _n_steps: int,
        _state: LocatingState,
    ) -> jax.Array:
        del _n_steps, _state
        k1, k2 = jax.random.split(key)
        i = jax.random.choice(k1, self.n, p=self.probs)
        mi, si = self.mean[i], self.stddev[i]
        return jax.random.normal(k2, shape=self.mean.shape[1:]) * si + mi


class LocUniform:
    def __init__(self, coordinate_or_list: Coordinate | list[float]) -> None:
        if isinstance(coordinate_or_list, list | tuple):
            self.coordinate = SquareCoordinate(
                tuple(coordinate_or_list[:2]),  # type: ignore
                tuple(coordinate_or_list[2:]),  # type: ignore
            )
        else:
            self.coordinate = coordinate_or_list

    def __call__(
        self,
        key: chex.PRNGKey,
        _n_steps: int,
        _state: LocatingState,
    ) -> jax.Array:
        del _n_steps, _state
        return self.coordinate.uniform(key)


class LocPeriodic:
    def __init__(self, *locations: ArrayLike) -> None:
        self._locations = jnp.array(locations)
        self._n = self._locations.shape[0]

    def __call__(
        self,
        _key: chex.PRNGKey,
        _n_steps: int,
        state: LocatingState,
    ) -> jax.Array:
        del _key, _n_steps
        return self._locations[state.n_produced % self._n]


class LocChoice:
    def __init__(self, *locations: ArrayLike) -> None:
        self._locations = jnp.array(locations)

    def __call__(
        self,
        key: chex.PRNGKey,
        _n_steps: int,
        _state: LocatingState,
    ) -> jax.Array:
        del _n_steps, _state
        return jax.random.choice(key, self._locations)


def _collect_loc_fn(fn_or_args: tuple[str, ...] | LocatingFn) -> LocatingFn:
    if callable(fn_or_args):
        return fn_or_args
    else:
        name, *init_args = fn_or_args
        fn, _ = Locating(name)(*init_args)
        return fn


def _collect_loc_fns(fns: Iterable[tuple[str, ...] | LocatingFn]) -> list[LocatingFn]:
    return [_collect_loc_fn(fn_or_args) for fn_or_args in fns]


class LocSwitching:
    """Branching based on how many foods are produced."""

    def __init__(
        self,
        interval: int,
        *loc_fns: tuple[str, ...] | LocatingFn,
    ) -> None:
        self._locfn_list = _collect_loc_fns(loc_fns)
        self._interval = interval
        self._n = len(self._locfn_list)

    def __call__(
        self,
        key: chex.PRNGKey,
        n_steps: int,
        state: LocatingState,
    ) -> jax.Array:
        index = (state.n_produced // self._interval) % self._n
        return jax.lax.switch(index, self._locfn_list, key, n_steps, state)


class LocScheduled:
    """Branching based on steps."""

    def __init__(
        self,
        intervals: int | list[int],
        *loc_fns: tuple[str, ...] | LocatingFn,
    ) -> None:
        self._locfn_list = _collect_loc_fns(loc_fns)
        if isinstance(intervals, int):
            intervals = [intervals * (i + 1) for i in range(len(self._locfn_list))]
        self._intervals = jnp.array(intervals, dtype=jnp.int32)

    def __call__(
        self,
        key: chex.PRNGKey,
        n_steps: int,
        state: LocatingState,
    ) -> jax.Array:
        index = jnp.digitize(n_steps, bins=self._intervals)
        return jax.lax.switch(index, self._locfn_list, key, n_steps, state)


def nth_true(boolean_array: jax.Array, n: int) -> jax.Array:
    return jnp.logical_and(boolean_array, jnp.cumsum(boolean_array) == n)


def first_to_nth_true(boolean_array: jax.Array, n: int | jax.Array) -> jax.Array:
    return jnp.logical_and(boolean_array, jnp.cumsum(boolean_array) <= n)


def place(
    n_trial: int,
    radius: float,
    coordinate: Coordinate,
    loc_fn: LocatingFn,
    loc_state: LocatingState,
    key: chex.PRNGKey,
    n_steps: int,
    shaped: ShapeDict,
    stated: StateDict,
) -> tuple[jax.Array, jax.Array]:
    keys = jax.random.split(key, n_trial)
    locations = jax.vmap(loc_fn, in_axes=(0, None, None))(keys, n_steps, loc_state)
    overlap = jax.vmap(circle_overlap, in_axes=(None, None, 0, None))(
        shaped,
        stated,
        locations,
        radius,
    )
    contains_fn = jax.vmap(coordinate.contains_circle, in_axes=(0, None))
    ok = jnp.logical_and(contains_fn(locations, radius), jnp.logical_not(overlap))
    mask = jnp.expand_dims(nth_true(ok, 1), axis=1)
    return jnp.sum(mask * locations, axis=0), jnp.any(ok)


@functools.partial(jax.vmap, in_axes=(0, None))
def _dist_mat(a: jax.Array, b: jax.Array) -> jax.Array:
    """Distance matrix between a and b"""
    return jnp.linalg.norm(jnp.expand_dims(a, axis=0) - b, axis=-1)


def place_multi(
    n_trial: int,
    n_max_placement: int,
    radius: float,
    coordinate: Coordinate,
    loc_fn: LocatingFn,
    loc_state: LocatingState,
    key: chex.PRNGKey,
    n_steps: int,
    shaped: ShapeDict,
    stated: StateDict,
) -> tuple[jax.Array, jax.Array]:
    keys = jax.random.split(key, n_trial)
    xy = jax.vmap(loc_fn, in_axes=(0, None, None))(keys, n_steps, loc_state)
    overlap = jax.vmap(circle_overlap, in_axes=(None, None, 0, None))(
        shaped,
        stated,
        xy,
        radius,
    )
    contains_fn = jax.vmap(coordinate.contains_circle, in_axes=(0, None))
    dm = _dist_mat(xy, xy)  # distance matrix of all generated points
    masked_dm = dm.at[jnp.tril_indices(n_trial)].set(jnp.inf)
    conflicts = ((masked_dm < 2.0 * radius).sum(axis=1)).astype(bool)
    ok = jnp.logical_and(contains_fn(xy, radius), jnp.logical_not(overlap | conflicts))
    return xy, first_to_nth_true(ok, n_max_placement)


def check_points_are_far_from_other_foods(
    min_dist_to_other_foods: float,
    index: int,
    xy: jax.Array,
    stated: StateDict,
) -> jax.Array:
    is_other = stated.static_circle.label != index
    is_active_other = jnp.logical_and(stated.static_circle.is_active, is_other)
    dm = _dist_mat(xy, stated.static_circle.p.xy)
    masked_dm = jnp.where(jnp.expand_dims(is_active_other, axis=0), dm, jnp.inf)
    return jnp.min(masked_dm, axis=-1) > min_dist_to_other_foods
