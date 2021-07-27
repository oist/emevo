"""
Waterworld environment.
Taken from `sisl`_ environment in PettingZoo.

Originaly developped by SISL (stanford intelligent system) and open-sourced as
as part of `MADRL`_ library, which is a modification of the single-agent version
that appears in `reinforcejs`_.

.. _sisl: https://www.pettingzoo.ml/sisl
.. _MADRL: https://github.com/sisl/MADRL
.. _reinforcejs: https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html
"""

import dataclasses
import itertools
import typing as t
import warnings

import numpy as np

from gym import spaces
from gym.utils import seeding
from scipy.stats import truncnorm
from scipy.spatial import distance as spd

from emevo.body import Body
from emevo.environment import Encount, Environment
from emevo.types import Info


Color = t.Tuple[int, int, int]
Self = t.Any


class Archea:
    def __init__(
        self,
        *,
        radius: float,
        max_accel: float,
    ) -> None:
        # Public members
        self.radius = radius
        self.max_accel = max_accel

        self._position = None
        self._velocity = None

    @property
    def position(self) -> np.ndarray:
        assert self._position is not None
        return self._position

    @position.setter
    def position(self, pos: np.ndarray) -> None:
        assert pos.shape == (2,)
        self._position = pos

    @property
    def velocity(self) -> np.ndarray:
        assert self._velocity is not None
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray) -> None:
        assert velocity.shape == (2,)
        self._velocity = velocity


class Pursuer(Archea, Body):
    def __init__(
        self,
        *,
        radius: float,
        max_accel: float,
        n_sensors: int,
        sensor_range: float,
        speed_features: bool = True,
        generation: int = 0,
    ) -> None:
        super().__init__(radius=radius, max_accel=max_accel)
        Body.__init__(self, name="Pursuer Archea", generation=generation)

        self.sensor_range = sensor_range

        # Number of observation coordinates from each sensor
        if speed_features:
            self._sensor_obs_coord = 8 * n_sensors
        else:
            self._sensor_obs_coord = 5 * n_sensors
        # +1 for is_colliding_evader, +1 for is_colliding_poison
        self._obs_dim = self._sensor_obs_coord + 2
        # Generate self._n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it
        # because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, n_sensors + 1)[:-1]
        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors

        self._action_space = spaces.Box(
            low=np.float32(-self.max_accel),
            high=np.float32(self.max_accel),
            shape=(2,),
            dtype=np.float32,
        )

        self._observation_space = spaces.Box(
            low=np.float32(-np.sqrt(2)),
            high=np.float32(2 * np.sqrt(2)),
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def sensors(self) -> np.ndarray:
        assert self._sensors is not None
        return self._sensors

    def sensed(
        self,
        object_coord: np.ndarray,
        object_radius: np.ndarray,
        same_idx: t.Optional[int] = None,
    ) -> np.ndarray:
        """Whether object would be sensed by the pursuers"""
        relative_coord = object_coord - np.expand_dims(self.position, 0)
        # Projection of object coordinate in direction of sensor
        sensorvals = self.sensors.dot(relative_coord.T)
        # Set sensorvals to np.inf when object should not be seen by sensor
        distance_squared = (relative_coord ** 2).sum(axis=1)[None, :]
        # Wrong direction (by more than 90 degrees in both directions)
        wrong_direction = sensorvals < 0
        # Outside sensor range
        outside = sensorvals - object_radius > self.sensor_range
        # Sensor does not intersect object
        does_not_intersect = distance_squared - sensorvals ** 2 > object_radius ** 2
        sensorvals[wrong_direction | outside | does_not_intersect] = np.inf
        # Set sensors values for sensing the current object to np.inf
        if same_idx is not None:
            sensorvals[:, same_idx] = np.inf
        return sensorvals

    def sense_barriers(self, min_pos: int = 0, max_pos: int = 1) -> np.ndarray:
        sensor_vectors = self.sensors * self.sensor_range
        sensor_endpoints = sensor_vectors + self.position

        # Clip sensor lines on the environment's barriers.
        # Note that any clipped vectors may not be
        # at the same angle as the original sensors
        clipped_endpoints = np.clip(sensor_endpoints, min_pos, max_pos)

        # Extract just the sensor vectors after clipping
        clipped_vectors = clipped_endpoints - self.position

        # Find the ratio of the clipped sensor vector to the original sensor vector
        # Scaling the vector by this ratio
        # will limit the end of the vector to the barriers
        ratios = np.divide(
            clipped_vectors,
            sensor_vectors,
            out=np.ones_like(clipped_vectors),
            where=np.abs(sensor_vectors) > 0.00000001,
        )

        # Find the minimum ratio (x or y) of clipped endpoints to original endpoints
        minimum_ratios = np.amin(ratios, axis=1)

        # Convert to 2d array of size (n_sensors, 1)
        sensor_values = np.expand_dims(minimum_ratios, 0)

        # Set values beyond sensor range to infinity
        does_sense = minimum_ratios < (1.0 - 1e-4)
        does_sense = np.expand_dims(does_sense, 0)
        sensor_values[np.logical_not(does_sense)] = np.inf

        # Convert -0 to 0
        sensor_values[sensor_values == -0] = 0

        return sensor_values.T


ReproduceFn = t.Callable[[t.List[Archea], np.random.RandomState], int]


def logistic_reproduce_fn(growth_rate: float, capacity: float) -> ReproduceFn:
    def reproduce_fn(archeas: t.List[Archea], _np_random: np.random.RandomState) -> int:
        n_archea = len(archeas)
        dn_dt = growth_rate * n_archea * (1 - n_archea / capacity)
        return max(0, int(dn_dt))

    return reproduce_fn


@dataclasses.dataclass(frozen=True)
class _Collision:
    distance_mat: np.ndarray
    collision_mat: np.ndarray
    caught_b: np.ndarray

    @staticmethod
    def empty() -> Self:
        return _Collision(np.array([]), np.array([[]]), np.array([], dtype=np.int))

    def listup(self, bodies: t.List[Archea]) -> t.List[Encount]:
        n_bodies = len(bodies)
        res = []
        for i in range(n_bodies):
            for j in range(i):
                if self.collision_mat[i, j]:
                    res.append(Encount((bodies[i], bodies[j]), self.distance_mat[i, j]))
        return res

    def n_caught(self, idx: int) -> int:
        return self.collision_mat[idx][self.caught_b].sum()

    def caught_archeas(self, archeas: t.List[Archea]) -> t.Iterable[Archea]:
        for caught_idx in self.caught_b:
            yield archeas[caught_idx]


def _remove_indices(list_: t.List[t.Any], indices: t.Iterable[t.Any]) -> t.List[t.Any]:
    if len(indices) == 0:
        return list_
    res = []
    for idx, elem in enumerate(list_):
        if idx not in indices:
            res.append(elem)
    return res


@dataclasses.dataclass(frozen=True)
class _Collisions:
    evader: _Collision
    poison: _Collision
    pursuer: _Collision


class WaterWorld(Environment):
    INFO_DESCRIPTIONS: t.ClassVar[t.Dict[str, str]] = {
        "food": "Number of foods the pursuer ate",
        "poison": "Number of poisons the pursuer ate",
    }

    def __init__(
        self,
        n_pursuers: int = 5,
        n_evaders: int = 5,
        n_poison: int = 10,
        n_required_pursuers: int = 1,
        n_sensors: int = 30,
        sensor_range: float = 0.2,
        pursuer_radius: float = 0.015,
        evader_radius_ratio: float = 2.0,
        poison_radius_ratio: float = 0.75,
        obstacle_radius: float = 0.2,
        obstacle_coords: t.Optional[np.ndarray] = np.array([0.5, 0.5]),
        n_obstacles: int = 1,
        pursuer_max_accel: float = 0.01,
        evader_max_accel: float = 0.01,
        poison_max_accel: float = 0.01,
        evader_reproduce_fn: ReproduceFn = logistic_reproduce_fn(1.0, 8),
        poison_reproduce_fn: ReproduceFn = logistic_reproduce_fn(1.0, 14),
        speed_features: bool = True,
        render_pixel_scale: int = 30 * 25,
        render_pursuers_with_mark: t.List[int] = [],
    ) -> None:
        """
        n_pursuers: number of pursuing archea (agents)
        n_evaders: number of evader archea
        n_poison: number of poison archea
        n_required_pursuers: number of pursuing archea (agents) that must be
                             touching food at the same time to consume it
                             For cooporative tasks, set this > 1
        n_sensors: number of sensors on all pursuing archea (agents)
        sensor_range: length of sensor dendrite on all pursuing archea (agents)
        pursuer_radius: radius of pursuers.
        evader_radius_ratio: ratio of evader's radius to pursuer's radious
        poison_radius_ratio: ratio of poison's radius to pursuer's radious
        obstacle_radius: radius of obstacle object
        obstacle_coord: coordinate of obstacle object.
                        Can be set to `None` to use a random location
        pursuer_max_accel: pursuer archea maximum acceleration (maximum action size)
        evader_max_accel: max accel of evader
        poison_max_accel: max accel of poison archea
        evader_reproduce_fn: reproducer of evader
        poison_reproduce_fn: reproducer of poison
        speed_features: toggles whether pursuing archea (agent) sensors
                        detect speed of other archea
        """
        self._initial_n_pursuers = n_pursuers
        self._initial_n_evaders = n_evaders
        self._initial_n_poison = n_poison
        self._n_pursuers = n_pursuers
        self._n_evaders = n_evaders
        self._n_poison = n_poison

        self._n_required_pursuers = n_required_pursuers
        self._obstacle_radius = obstacle_radius
        if obstacle_coords is not None:
            if obstacle_coords.ndim > 2 or obstacle_coords.shape[-1] != 2:
                raise ValueError(
                    f"Invalid shape as coordinates: {obstacle_coord.shape}"
                )
            if obstacle_coords.ndim == 1:
                obstacle_coords = np.expand_dims(obstacle_coords, 0)
            if obstacle_coords.shape[0] != n_obstacles:
                raise ValueError(
                    f"n_obstacles = {n_obstacles}, but obstacles "
                    + f"with shape {obstacle_coords.shape} is given"
                )
        self._initial_obstacle_coords = obstacle_coords
        self._obstacle_coords: t.Optional[np.ndarray] = None
        self._pursuer_radius = pursuer_radius
        self._evader_radius = pursuer_radius * evader_radius_ratio
        self._poison_radius = pursuer_radius * poison_radius_ratio
        self._n_sensors = n_sensors
        self._n_obstacles = n_obstacles
        self._speed_features = speed_features
        self._pursuer_sensor_range = sensor_range

        def _generate_pursuer(generation: int = 0) -> Pursuer:
            return Pursuer(
                radius=self._pursuer_radius,
                n_sensors=self._n_sensors,
                max_accel=pursuer_max_accel,
                sensor_range=self._pursuer_sensor_range,
                speed_features=self._speed_features,
                generation=generation,
            )

        self._generate_pursuer = _generate_pursuer
        self._generate_evader = lambda: Archea(
            radius=self._evader_radius, max_accel=evader_max_accel
        )
        self._generate_poison = lambda: Archea(
            radius=self._poison_radius, max_accel=poison_max_accel
        )
        self._pursuers = [self._generate_pursuer() for _ in range(self._n_pursuers)]
        self._evaders = [self._generate_evader() for _ in range(self._n_evaders)]
        self._poisons = [self._generate_poison() for _ in range(self._n_poison)]

        # Observational informations for each agent
        self._last_observations: t.List[t.Optional[np.ndarray]] = [
            None for _ in range(self._n_pursuers)
        ]
        self._last_collisions: t.Optional[_Collisions] = None
        self._consumed_energy = [0.0 for _ in range(self._n_pursuers)]

        self._unit_time = 1.0
        self._n_steps = 0

        # reproducers
        self._evader_reproduce_fn = evader_reproduce_fn
        self._poison_reproduce_fn = poison_reproduce_fn

        # Visualization stuffs
        self.pursuers_with_mark = render_pursuers_with_mark
        self._pixel_scale = render_pixel_scale
        self._viewer = None

        # Call seed and reset for convenience
        self.seed()
        self.reset()

    # Methods required by Environment

    def act(self, pursuer: Pursuer, action: np.ndarray) -> None:
        action = np.asarray(action).reshape(2)
        speed = np.linalg.norm(action)
        if speed > pursuer.max_accel:
            # Limit added thrust to pursuer.max_accel
            action = action / speed * pursuer.max_accel

        pursuer.velocity = pursuer.velocity + action
        pursuer.position += self._unit_time * pursuer.velocity

        self._consumed_energy[self._idx(pursuer)] = np.linalg.norm(action)

    def available_bodies(self) -> t.Iterable[Pursuer]:
        return iter(self._pursuers)

    def step(self) -> t.List[Encount]:
        def move_archeas(archeas: t.List[Archea]) -> None:
            for archea in archeas:
                # Move archeaects
                archea.position += self._unit_time * archea.velocity
                # Bounce archeaect if it hits a wall
                for i in range(len(archea.position)):
                    if archea.position[i] >= 1 or archea.position[i] <= 0:
                        archea.position[i] = np.clip(archea.position[i], 0, 1)
                        archea.velocity[i] = -1 * archea.velocity[i]

        move_archeas(self._evaders)
        move_archeas(self._poisons)

        self._n_steps += 1
        if self._n_pursuers > 0:
            self._last_observations = self._collision_handling_impl()
            self._reproduce_archeas()
            return self._last_collisions.pursuer.listup(self._pursuers)
        else:
            warnings.warn("step is called after pursuers are distinct!")
            self._collision_handling_impl()
            return []

    def observe(self, body: Body) -> t.Optional[t.Tuple[np.ndarray, Info]]:
        idx = self._idx(body)
        obs = self._last_observations[idx]
        if obs is None:
            # If obs is None, then the agent is a newborn and have observed nothing.
            return None
        info = {
            "food": self._last_collisions.evader.n_caught(idx),
            "poison": self._last_collisions.poison.n_caught(idx),
        }
        return obs, info

    def born(self, generation: int = 0, place: t.Optional[np.ndarray] = None) -> Body:
        if place is not None:
            if any(map(lambda p: p < 0 or 1 < p, place)):
                warnings.warn(f"Place {place} is out of the field")
                place = np.clip(place, 0.0, 1.0)
        body = self._generate_pursuer(generation)
        self._initialize_archea(body, position=place, velocity=np.zeros(2))
        self._maybe_rebound_archea(body)
        self._pursuers.append(body)
        self._consumed_energy.append(0.0)
        self._last_observations.append(None)
        self._n_pursuers += 1
        # Since we don't touch collisions when obs is None, it's OK to do nothing
        return body

    def die(self, body: Pursuer) -> bool:
        idx = self._idx(body)
        self._pursuers.pop(idx)
        self._consumed_energy.pop(idx)
        self._last_observations.pop(idx)
        self._n_pursuers -= 1
        return self._n_pursuers == 0

    def reset(self) -> None:
        """
        Reset the state and returns the observation of the first agent.
        """

        self._n_steps = 0
        # Initialize obstacles
        if self._initial_obstacle_coords is None:
            # Generate xsxfobstacle positions in range [0, 1)
            self._obstacle_coords = self._np_random.rand(self._n_obstacles, 2)
        else:
            self._obstacle_coords = self._initial_obstacle_coords.copy()

        def maybe_reset_archeas(
            archeas: t.List[Archea],
            initial_n_archeas: int,
            generater: t.Callable[[], Archea],
        ) -> t.List[Archea]:
            if len(archeas) != initial_n_archeas:
                return [generater() for _ in range(initial_n_archeas)]
            else:
                return archeas

        # Initialize pursuers
        self._pursuers = maybe_reset_archeas(
            self._pursuers, self._initial_n_pursuers, self._generate_pursuer
        )
        for pursuer in self._pursuers:
            self._initialize_archea(pursuer, velocity=np.zeros(2))

        # Initialize evaders and poisons
        self._evaders = maybe_reset_archeas(
            self._evaders, self._initial_n_evaders, self._generate_evader
        )
        self._poisons = maybe_reset_archeas(
            self._poisons, self._initial_n_poison, self._generate_poison
        )
        for archea in itertools.chain(self._evaders, self._poisons):
            self._initialize_archea(archea)

        self._last_observations = self._collision_handling_impl()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    def seed(self, seed: t.Optional[int] = None) -> int:
        self._np_random, seed = seeding.np_random(seed)
        return seed

    @property
    def np_random(self) -> t.Optional[np.random.RandomState]:
        return self._np_random

    def render(self, mode: str = "human") -> t.Union[None, np.ndarray]:
        if self._viewer is None:
            try:
                import pygame
            except ImportError as e:
                raise ImportError(
                    "To render waterworld, you need to install pygame by"
                    + " e.g. pip install pygame"
                ) from e

            self._viewer = _Viewer(mode, self._pixel_scale, pygame)

        self._viewer.draw_background()
        self._viewer.draw_obstacles(self._obstacle_coords, self._obstacle_radius)
        self._viewer.draw_archeas(self._pursuers, "pursuer")
        self._viewer.draw_archeas(self._evaders, "evader")
        self._viewer.draw_archeas(self._poisons, "poison")

        for idx in self.pursuers_with_mark:
            self._viewer.mark_archea(self._pursuers[idx], (255, 0, 255))

        if mode == "human":
            self._viewer.pygame.display.flip()
        else:
            observation = self._viewer.pygame.surfarray.pixels3d(self.screen)
            return np.transpose(observation.copy(), axes=(1, 0, 2))

    # Other methods

    def _idx(self, pursuer: Pursuer) -> int:
        try:
            return self._pursuers.index(pursuer)
        except ValueError as e:
            raise ValueError(f"Invalid pursuer: {pursuer}") from e

    def _generate_poison(self) -> Archea:
        return Archea(
            radius=self._poison_radius,
            max_accel=self._poison_max_accel,
        )

    def _initialize_archea(
        self,
        archea: Archea,
        *,
        position: t.Optional[np.ndarray] = None,
        velocity: t.Optional[np.ndarray] = None,
    ) -> None:
        if position is None:
            position = self._np_random.rand(2)
            while self._detect_collision_to_obs(archea.radius, position) is not None:
                position = self._np_random.rand(2)
            rebound = False
        else:
            rebound = True
        if velocity is None:
            velocity = self._sample_velocity(archea.max_accel)
        archea.position = position
        archea.velocity = velocity
        if rebound:
            self._maybe_rebound_archea(archea)

    def _detect_collision_to_obs(
        self,
        radius: float,
        coord: np.ndarray,
    ) -> t.Optional[int]:
        """Return index of the obstalce if the archea collides"""
        dists = spd.cdist(coord.reshape(1, 2), self._obstacle_coords).ravel()
        (collision_indices,) = np.where(dists < radius + self._obstacle_radius)
        n_collisions = len(collision_indices)
        if n_collisions == 0:
            return None
        else:
            assert n_collisions == 1, (
                "Multiple collision to obstacles are detected.\n"
                + "Please place obstacles more sparsely"
            )
            return collision_indices.item()

    def _sample_velocity(self, max_norm: float) -> np.ndarray:
        """Sample velocity from two independent trunnorms and then normalize it"""
        velocity = truncnorm.ppf(self._np_random.rand(2), -max_norm, max_norm)
        speed = np.linalg.norm(velocity)
        if speed > max_norm:
            # Limit speed
            return velocity / speed * max_norm
        else:
            return velocity

    def _detect_collision(
        self,
        positions_a: np.ndarray,
        positions_b: np.ndarray,
        threshold: float,
        *,
        n_required_collisions: int = 1,
        a_equal_to_b: bool = False,
    ) -> t.Optional[_Collision]:
        """
        Detect collision and check whether it results in catching the object.
        This is because you need `n_required_pursuers` agents to collide
        with the object to actually catch it.
        """
        alen, blen = len(positions_a), len(positions_b)
        if alen == 0 or blen == 0:
            return _Collision(
                np.empty([alen, blen]),
                np.empty([alen, blen]),
                np.array([], dtype=np.int64),
            )

        if positions_a.ndim == 1:
            positions_a = np.expand_dims(positions_a, axis=0)
        if positions_b.ndim == 1:
            positions_b = np.expand_dims(positions_b, axis=0)

        distances = spd.cdist(positions_a, positions_b)
        is_colliding_a_b = distances <= threshold
        if a_equal_to_b:
            indices = np.arange(alen)
            is_colliding_a_b[indices, indices] = False

        # Number of collisions for each y
        n_collisions_b = is_colliding_a_b.sum(axis=0)
        # List of b that have been caught
        caught_b = np.where(n_collisions_b >= n_required_collisions)[0]
        return _Collision(distances, is_colliding_a_b, caught_b)

    def _closest_dist(
        self,
        closest_object_idx: np.ndarray,
        input_sensorvals: np.ndarray,
    ) -> np.ndarray:
        """Closest distances according to `idx`"""
        sensorvals = []

        for pursuer_idx in range(self._n_pursuers):
            sensors = np.arange(self._n_sensors)  # sensor indices
            objects = closest_object_idx[pursuer_idx, ...]  # object indices
            sensorvals.append(input_sensorvals[pursuer_idx, ..., sensors, objects])

        return np.c_[sensorvals]

    def _extract_speed_features(
        self,
        object_velocities: np.ndarray,
        object_sensorvals: np.ndarray,
        sensed_mask: np.ndarray,
    ) -> np.ndarray:
        """
        object_velocities: velocities of objected archeas
        object_sensorvals: sensor values of objected archeas
        sensed_mask: a boolean mask of which sensor values detected an object
        """
        speed_features = np.zeros((self._n_pursuers, self._n_sensors))
        if len(object_velocities) == 0:
            return speed_features

        sensorvals = []
        for pursuer in self._pursuers:
            relative_speed = object_velocities - np.expand_dims(pursuer.velocity, 0)
            sensorvals.append(pursuer.sensors.dot(relative_speed.T))
        sensed_speed = np.c_[sensorvals]  # Speeds in direction of each sensor

        sensorvals = []
        for pursuer_idx in range(self._n_pursuers):
            sensorvals.append(
                sensed_speed[pursuer_idx, :, :][
                    np.arange(self._n_sensors), object_sensorvals[pursuer_idx, :]
                ]
            )
        # Set sensed values, all others remain 0
        speed_features[sensed_mask] = np.c_[sensorvals][sensed_mask]

        return speed_features

    def _rebound_archea(self, obstacle_coord: np.ndarray, archea: Archea) -> None:
        center_to_archea = archea.position - obstacle_coord
        ctoa_norm = np.linalg.norm(center_to_archea)
        # ratio of the vector from center to archea
        ratio_of_ca = ctoa_norm / (self._obstacle_radius + archea.radius)
        # ratio of the vector from archea to edge
        ratio_of_ae = 1.0 - ratio_of_ca
        new_pos = archea.position + center_to_archea * (ratio_of_ae / ratio_of_ca) * 2.0
        archea.position = new_pos

        # project current velocity onto collision normal
        current_vel = archea.velocity
        collision_normal = new_pos - obstacle_coord
        proj_numer = np.dot(current_vel, collision_normal)
        cllsn_mag = np.dot(collision_normal, collision_normal)
        proj_vel = (proj_numer / cllsn_mag) * collision_normal
        perp_vel = current_vel - proj_vel
        total_vel = perp_vel - proj_vel
        archea.velocity = total_vel

    def _reproduce_archeas(self) -> None:
        n_new_evaders = self._evader_reproduce_fn(self._evaders, self._np_random)
        for _ in range(n_new_evaders):
            new_evader = self._generate_evader()
            self._initialize_archea(new_evader)
            self._evaders.append(new_evader)

        n_new_poison = self._poison_reproduce_fn(self._poisons, self._np_random)
        for _ in range(n_new_poison):
            new_poison = self._generate_poison()
            self._initialize_archea(new_poison)
            self._poisons.append(new_poison)

        self._n_evaders += n_new_evaders
        self._n_poison += n_new_poison

    def _maybe_rebound_archea(self, archea: Archea) -> None:
        collision_idx = self._detect_collision_to_obs(archea.radius, archea.position)
        if collision_idx is not None:
            self._rebound_archea(self._obstacle_coords[collision_idx], archea)

    def _collision_handling_impl(self) -> t.List[np.ndarray]:
        # assert self._n_evaders > 0 and self._n_poison > 0

        # Stop pursuers upon hitting a wall
        for pursuer in self._pursuers:
            clipped_coord = np.clip(pursuer.position, 0, 1)
            clipped_velocity = pursuer.velocity
            # If x or y position gets clipped, set x or y velocity to 0 respectively
            clipped_velocity[pursuer.position != clipped_coord] = 0
            # Save clipped velocity and position
            pursuer.velocity = clipped_velocity
            pursuer.position = clipped_coord

        # Rebound an archea if it hits an obstacle
        for archea in itertools.chain(self._pursuers, self._evaders, self._poisons):
            self._maybe_rebound_archea(archea)

        positions_pursuer = np.array([pursuer.position for pursuer in self._pursuers])
        positions_evader = np.array([evader.position for evader in self._evaders])
        positions_poison = np.array([poison.position for poison in self._poisons])

        # Find evader collisions
        evader_collision = self._detect_collision(
            positions_pursuer,
            positions_evader,
            self._pursuer_radius + self._evader_radius,
            n_required_collisions=self._n_required_pursuers,
        )

        # Find poison collisions
        poison_collision = self._detect_collision(
            positions_pursuer,
            positions_poison,
            self._pursuer_radius + self._poison_radius,
        )

        # Find pursuer collisions
        pursuer_collision = self._detect_collision(
            positions_pursuer,
            positions_pursuer,
            self._pursuer_radius * 2,
            a_equal_to_b=True,
        )

        # Find sensed obstacles
        sensorvals_pursuer_obstacle = [
            pursuer.sensed(self._obstacle_coords, self._obstacle_radius)
            for pursuer in self._pursuers
        ]
        # Find sensed barriers
        sensorvals_pursuer_barrier = [
            pursuer.sense_barriers() for pursuer in self._pursuers
        ]

        # Find sensed evaders
        if self._n_evaders > 0:
            sensorvals_pursuer_evader = [
                pursuer.sensed(positions_evader, self._evader_radius)
                for pursuer in self._pursuers
            ]
        else:
            sensorvals_pursuer_evader = []

        # Find sensed poisons
        if self._n_poison > 0:
            sensorvals_pursuer_poison = [
                pursuer.sensed(positions_poison, self._poison_radius)
                for pursuer in self._pursuers
            ]
        else:
            sensorvals_pursuer_poison = []

        # Find sensed pursuers
        sensorvals_pursuer_pursuer = [
            self._pursuers[idx].sensed(
                positions_pursuer, self._pursuer_radius, same_idx=idx
            )
            for idx in range(self._n_pursuers)
        ]

        # Collect distance features
        def sensor_features(
            sensorvals: t.List[np.ndarray],
        ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if len(sensorvals) == 0:
                empty = np.array([], dtype=np.int64)
                return empty, empty, empty
            sensorvals = np.stack(sensorvals, axis=0)
            closest_idx_array = np.argmin(sensorvals, axis=2)
            closest_distances = self._closest_dist(closest_idx_array, sensorvals)
            finite_mask = np.isfinite(closest_distances)
            sensed_distances = np.ones((self._n_pursuers, self._n_sensors))
            sensed_distances[finite_mask] = closest_distances[finite_mask]
            return sensed_distances, closest_idx_array, finite_mask

        obstacle_distance_features, _, _ = sensor_features(sensorvals_pursuer_obstacle)
        barrier_distance_features, _, _ = sensor_features(sensorvals_pursuer_barrier)
        evader_distance_features, closest_evader_idx, evader_mask = sensor_features(
            sensorvals_pursuer_evader
        )
        poison_distance_features, closest_poison_idx, poison_mask = sensor_features(
            sensorvals_pursuer_poison
        )
        pursuer_distance_features, closest_pursuer_idx, pursuer_mask = sensor_features(
            sensorvals_pursuer_pursuer
        )

        # Memonize collisions
        self._last_collisions = _Collisions(
            evader_collision,
            poison_collision,
            pursuer_collision,
        )

        # Add features together
        if self._speed_features:
            evader_speed_features = self._extract_speed_features(
                np.array([evader.velocity for evader in self._evaders]),
                closest_evader_idx,
                evader_mask,
            )
            poison_speed_features = self._extract_speed_features(
                np.array([poison.velocity for poison in self._poisons]),
                closest_poison_idx,
                poison_mask,
            )
            pursuer_speed_features = self._extract_speed_features(
                np.array([pursuer.velocity for pursuer in self._pursuers]),
                closest_pursuer_idx,
                pursuer_mask,
            )

            all_features = [
                obstacle_distance_features,
                barrier_distance_features,
                evader_distance_features,
                evader_speed_features,
                poison_distance_features,
                poison_speed_features,
                pursuer_distance_features,
                pursuer_speed_features,
            ]
        else:
            all_features = [
                obstacle_distance_features,
                barrier_distance_features,
                evader_distance_features,
                poison_distance_features,
                pursuer_distance_features,
            ]

        obs_list = []
        nonempty_features = list(filter(lambda arr: len(arr) > 0, all_features))
        if len(nonempty_features) > 0:
            sensorfeatures = np.column_stack(nonempty_features)
            has_collided_to_ev = evader_collision.collision_mat.sum(axis=1) > 0
            has_collided_to_po = poison_collision.collision_mat.sum(axis=1) > 0
            for pursuer_idx in range(self._n_pursuers):
                obs = [
                    sensorfeatures[pursuer_idx].ravel(),
                    [has_collided_to_ev[pursuer_idx], has_collided_to_po[pursuer_idx]],
                ]
                obs_list.append(np.concatenate(obs))

        # Remove caught archeas
        self._evaders = _remove_indices(self._evaders, evader_collision.caught_b)
        self._poisons = _remove_indices(self._poisons, poison_collision.caught_b)
        self._n_evaders, self._n_poison = len(self._evaders), len(self._poisons)

        return obs_list


class _Viewer:
    """Visualizer of Waterworld using pygame"""

    _BLACK: Color = 0, 0, 0
    _COLORS: t.Dict[str, Color] = {
        "pursuer": (101, 104, 209),
        "evader": (238, 116, 106),
        "poison": (145, 250, 116),
    }
    _OBSTACLE_GREEN: Color = 120, 176, 178
    _WHITE: Color = (255, 255, 255)

    def __init__(self, mode: str, pixel_scale: int, pygame: "module") -> None:
        self.pygame = pygame

        self._pixel_scale = pixel_scale
        self._xoffset = int(pixel_scale * 0.4)
        self._screen_size = self._pixel_scale + self._xoffset, self._pixel_scale

        if mode == "human":
            pygame.display.init()
            self._screen = pygame.display.set_mode(self._screen_size)
        elif mode == "rgb-array":
            self._screen = pygame.Surface(self._screen_size)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self._archea_names = list(self._COLORS.keys())
        pygame.font.init()
        self._font = pygame.font.SysFont(None, 24)

    def close(self) -> None:
        self.pygame.display.quit()
        self.pygame.quit()

    def mark_archea(self, archea: Archea, color: Color) -> None:
        x, y = archea.position
        center = int(self._pixel_scale * x), int(self._pixel_scale * y)
        self.pygame.draw.circle(
            self._screen,
            color,
            center,
            self._pixel_scale * archea.radius * 0.25,
        )

    def draw_archeas(self, archeas: t.List[Archea], name: str) -> None:
        n_archeas = len(archeas)
        if n_archeas == 0:
            return

        for archea in archeas:
            x, y = archea.position
            center = int(self._pixel_scale * x), int(self._pixel_scale * y)
            if isinstance(archea, Pursuer):
                for sensor in archea._sensors:
                    start = center
                    end = center + self._pixel_scale * (archea.sensor_range * sensor)
                    self.pygame.draw.line(self._screen, self._BLACK, start, end, 1)
            self.pygame.draw.circle(
                self._screen,
                self._COLORS[name],
                center,
                self._pixel_scale * archea.radius,
            )

        x_start = self._pixel_scale + self._xoffset // 2
        y_position = (self._archea_names.index(name) + 4) * (self._pixel_scale // 7)
        self.pygame.draw.circle(
            self._screen,
            self._COLORS[name],
            (x_start, y_position),
            self._pixel_scale * archea.radius,
        )
        img = self._font.render(f": {n_archeas}", True, self._BLACK)
        self._screen.blit(img, (x_start + self._xoffset // 4, y_position))

    def draw_background(self) -> None:
        # -1 is building pixel flag
        rect = self.pygame.Rect(0, 0, *self._screen_size)
        self.pygame.draw.rect(self._screen, self._WHITE, rect)

    def draw_obstacles(self, obstacle_coords: np.ndarray, radius: float) -> None:
        for obstacle in obstacle_coords:
            x, y = obstacle
            center = int(self._pixel_scale * x), int(self._pixel_scale * y)
            self.pygame.draw.circle(
                self._screen,
                self._OBSTACLE_GREEN,
                center,
                self._pixel_scale * radius,
            )
