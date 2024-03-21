"""Place agent and food"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable
from typing import Any, Callable, Protocol, cast

import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from emevo.environments.phyjax2d import ShapeDict, StateDict
from emevo.environments.phyjax2d_utils import circle_overlap

Self = Any


@chex.dataclass
class FoodNumState:
    current: jax.Array
    internal: jax.Array

    def appears(self) -> jax.Array:
        return (self.internal - self.current) >= 1.0

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
        return jax.tree_map(lambda x: x[index], self)


class ReprNumFn(Protocol):
    initial: int

    def __call__(self, n_steps: int, state: FoodNumState) -> FoodNumState: ...


@dataclasses.dataclass(frozen=True)
class ReprNumConstant:
    initial: int

    def __call__(self, _: int, state: FoodNumState) -> FoodNumState:
        # Do nothing here
        return state._update(jnp.array(self.initial, dtype=jnp.float32))


@dataclasses.dataclass(frozen=True)
class ReprNumLinear:
    initial: int
    dn_dt: float

    def __call__(self, _: int, state: FoodNumState) -> FoodNumState:
        # Increase the number of foods by dn_dt
        internal = jnp.fmax(state.current, state.internal)
        max_value = jnp.array(self.initial, dtype=jnp.float32)
        return state._update(jnp.clip(internal + self.dn_dt, a_max=max_value))


@dataclasses.dataclass(frozen=True)
class ReprNumLogistic:
    initial: int
    growth_rate: float
    capacity: float

    def __call__(self, _: int, state: FoodNumState) -> FoodNumState:
        internal = jnp.fmax(state.current, state.internal)
        dn_dt = self.growth_rate * internal * (1 - internal / self.capacity)
        return state._update(internal + dn_dt)


class ReprNumCycle:
    def __init__(
        self,
        interval: int,
        *num_fns: tuple[str, ...] | ReprNumFn,
    ) -> None:
        numfn_list = []
        for fn_or_base in num_fns:
            if callable(fn_or_base):
                numfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                fn, _ = ReprNum(name)(*args)
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


class ReprNumScheduled:
    """Branching based on steps."""

    def __init__(
        self,
        intervals: int | list[int],
        *num_fns: tuple[str, ...] | ReprNumFn,
    ) -> None:
        numfn_list = []
        for fn_or_base in num_fns:
            if callable(fn_or_base):
                numfn_list.append(fn_or_base)
            else:
                name, *args = fn_or_base
                fn, _ = ReprNum(name)(*args)
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


class ReprNum(str, enum.Enum):
    """Methods to determine the number of foods reproduced."""

    CONSTANT = "constant"
    CYCLE = "cycle"
    LINEAR = "linear"
    LOGISTIC = "logistic"
    SCHEDULED = "scheduled"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[ReprNumFn, FoodNumState]:
        if self is ReprNum.CONSTANT:
            fn = ReprNumConstant(*args, **kwargs)
        elif self is ReprNum.CYCLE:
            fn = ReprNumCycle(*args, **kwargs)
        elif self is ReprNum.LINEAR:
            fn = ReprNumLinear(*args, **kwargs)
        elif self is ReprNum.LOGISTIC:
            fn = ReprNumLogistic(*args, **kwargs)
        elif self is ReprNum.SCHEDULED:
            fn = ReprNumScheduled(*args, **kwargs)
        else:
            raise AssertionError("Unreachable")

        initial = fn.initial
        state = FoodNumState(
            current=jnp.array(int(initial), dtype=jnp.int32),
            internal=jnp.array(float(initial), dtype=jnp.float32),
        )
        return cast(ReprNumFn, fn), state


class Coordinate(Protocol):
    def bbox(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

    def contains_circle(
        self, center: jax.Array, radius: jax.Array | float
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
        return jax.tree_map(lambda x: x[index], self)


LocatingFn = Callable[[chex.PRNGKey, int, LocatingState], jax.Array]


class Locating(str, enum.Enum):
    """Methods to determine the location of new foods or agents"""

    GAUSSIAN = "gaussian"
    GAUSSIAN_MIXTURE = "gaussian-mixture"
    PERIODIC = "periodic"
    SCHEDULED = "scheduled"
    SWITCHING = "switching"
    UNIFORM = "uniform"

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[LocatingFn, LocatingState]:
        # Make sure it has shape () because it's also used for agents as singleton
        state = LocatingState(
            n_produced=jnp.array(0, dtype=jnp.int32),
        )
        if self is Locating.GAUSSIAN:
            return loc_gaussian(*args, **kwargs), state
        elif self is Locating.GAUSSIAN_MIXTURE:
            return loc_gaussian_mixture(*args, **kwargs), state
        elif self is Locating.PERIODIC:
            return LocPeriodic(*args, **kwargs), state
        elif self is Locating.UNIFORM:
            return loc_uniform(*args, **kwargs), state
        elif self is Locating.SCHEDULED:
            return LocScheduled(*args, **kwargs), state
        elif self is Locating.SWITCHING:
            return LocSwitching(*args, **kwargs), state
        else:
            raise AssertionError("Unreachable")


def loc_gaussian(mean: ArrayLike, stddev: ArrayLike) -> LocatingFn:
    mean_a = jnp.array(mean)
    std_a = jnp.array(stddev)
    shape = mean_a.shape

    def sample(key: chex.PRNGKey, _n_steps: int, _state: LocatingState) -> jax.Array:
        del _n_steps, _state
        return jax.random.normal(key, shape=shape) * std_a + mean_a

    return sample


def loc_gaussian_mixture(
    probs: ArrayLike,
    mean_arr: ArrayLike,
    stddev_arr: ArrayLike,
) -> LocatingFn:
    mean_a = jnp.array(mean_arr)
    stddev_a = jnp.array(stddev_arr)
    probs_a = jnp.array(probs)
    n = probs_a.shape[0]

    def sample(key: chex.PRNGKey, _n_steps: int, _state: LocatingState) -> jax.Array:
        del _n_steps, _state
        k1, k2 = jax.random.split(key)
        i = jax.random.choice(k1, n, p=probs_a)
        mi, si = mean_a[i], stddev_a[i]
        return jax.random.normal(k2, shape=mean_a.shape[1:]) * si + mi

    return sample


def loc_uniform(coordinate: Coordinate) -> LocatingFn:
    def sample(key: chex.PRNGKey, _n_steps: int, _state: LocatingState) -> jax.Array:
        del _n_steps, _state
        return coordinate.uniform(key)

    return sample


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


def _collect_loc_fns(fns: Iterable[tuple[str, ...] | LocatingFn]) -> list[LocatingFn]:
    locfn_list = []
    for fn_or_args in fns:
        if callable(fn_or_args):
            locfn_list.append(fn_or_args)
        else:
            name, *init_args = fn_or_args
            fn, _ = Locating(name)(*init_args)
            locfn_list.append(fn)
    return locfn_list


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
