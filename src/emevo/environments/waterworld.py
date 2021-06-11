"""
Waterworld environment.
Taken from `sisl`_ environment in PettingZoo.

Originaly developped by SISL (stanford intelligent system) and open-sourced as
as part of `MADRL`_ library.

.. _sisl: https://www.pettingzoo.ml/sisl
.. _MADRL: https://github.com/sisl/MADRL
"""

import typing as t

import numpy as np

from gym import spaces
from gym.utils import seeding
from scipy.spatial import distance as spd

from emevo.body import Body


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

    @property
    def velocity(self) -> np.ndarray:
        assert self._velocity is not None
        return self._velocity

    def set_position(self, pos: np.ndarray) -> None:
        assert pos.shape == (2,)
        self._position = pos

    def set_velocity(self, velocity: np.ndarray) -> None:
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

    def is_dead(self) -> bool:
        return False

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


class WaterWorldEvent:
    pass


class WaterWorld:
    def __init__(
        self,
        n_pursuers: int = 5,
        n_evaders: int = 5,
        n_poison: int = 10,
        n_coop: int = 2,
        n_sensors: int = 30,
        sensor_range: float = 0.2,
        pursuer_radius: float = 0.015,
        evader_radius_ratio: float = 2.0,
        poison_radius_ratio: float = 0.75,
        obstacle_radius: float = 0.2,
        obstacle_coords: t.Optional[np.ndarray] = np.array([0.5, 0.5]),
        n_obstacles: int = 1,
        pursuer_max_accel: float = 0.01,
        evader_max_speed: float = 0.01,
        poison_max_speed: float = 0.01,
        poison_reward: float = -1.0,
        food_reward: float = 10.0,
        encounter_reward: float = 0.01,
        thrust_penalty: float = -0.5,
        speed_features: bool = True,
        max_cycles: int = 500,
    ) -> None:
        """
        n_pursuers: number of pursuing archea (agents)
        n_evaders: number of evader archea
        n_poison: number of poison archea
        n_coop: number of pursuing archea (agents) that must be touching food
                at the same time to consume it
        n_sensors: number of sensors on all pursuing archea (agents)
        sensor_range: length of sensor dendrite on all pursuing archea (agents)
        pursuer_radius: radius of pursuers.
        evader_radius_ratio: ratio of evader's radius to pursuer's radious
        poison_radius_ratio: ratio of poison's radius to pursuer's radious
        obstacle_radius: radius of obstacle object
        obstacle_coord: coordinate of obstacle object.
                        Can be set to `None` to use a random location
        pursuer_max_accel: pursuer archea maximum acceleration (maximum action size)
        evader_max_speed: max speed of evader
        poison_max_speed: max spped of poison archea
        poison_reward: reward for pursuer consuming a poison object (typically negative)
        food_reward: reward for pursuers consuming an evading archea
        encounter_reward: reward for a pursuer colliding with an evading archea
        thrust_penalty: scaling factor for the negative reward
                        used to penalize large actions
        speed_features: toggles whether pursuing archea (agent) sensors
                        detect speed of other archea
        max_cycles: After max_cycles steps all agents will return done
        """
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_poison = n_poison
        self.n_coop = n_coop
        self.obstacle_radius = obstacle_radius
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
        self.initial_obstacle_coords = obstacle_coords
        self.pursuer_max_accel = pursuer_max_accel
        self.pursuer_radius = pursuer_radius
        self.evader_radius = pursuer_radius * evader_radius_ratio
        self.poison_radius = pursuer_radius * poison_radius_ratio
        self.n_sensors = n_sensors
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.thrust_penalty = thrust_penalty
        self.encounter_reward = encounter_reward
        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_obs = [None for _ in range(self.n_pursuers)]

        self.n_obstacles = n_obstacles
        self._speed_features = speed_features
        self.max_cycles = max_cycles
        self.seed()
        self.pursuers = set()
        self._pursuers = [
            Pursuer(
                radius=self.pursuer_radius,
                n_sensors=self.n_sensors,
                max_accel=self.pursuer_max_accel,
                sensor_range=sensor_range,
                speed_features=self._speed_features,
            )
            for _ in range(self.n_pursuers)
        ]
        self._evaders = [
            Archea(
                radius=self.evader_radius,
                max_accel=evader_max_speed,
            )
            for _ in range(self.n_evaders)
        ]
        self._poisons = [
            Archea(
                radius=self.poison_radius,
                max_accel=poison_max_speed,
            )
            for _ in range(self.n_poison)
        ]

        self.pixel_scale = 30 * 25

        self.cycle_time = 1.0
        self.frames = 0
        self._viewer = None
        self.reset()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    def seed(self, seed: t.Optional[int] = None) -> t.List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _generate_coord(self, radius: float) -> np.ndarray:
        coord = self.np_random.rand(2)
        collide = radius * 2 + self.obstacle_radius
        # Create random coordinate that avoids obstacles
        while spd.cdist(coord.reshape(1, 2), self.obstacle_coords).max() <= collide:
            coord = self.np_random.rand(2)
        return coord

    def _sample_velocity(self, max_norm: float) -> np.ndarray:
        """Sample velocity from standard normal distribution and
        truncate it if necessary.
        """
        velocity = self.np_random.rand(2) - 0.5
        speed = np.linalg.norm(velocity)
        if speed > max_norm:
            # Limit speed
            return velocity / speed * max_norm
        else:
            return velocity

    def place_pursuer(self) -> None:
        pass

    def reset(self) -> None:
        """
        Reset the state and returns the observation of the first agent.
        """

        self.frames = 0
        # Initialize obstacles
        if self.initial_obstacle_coords is None:
            # Generate obstacle positions in range [0, 1)
            self.obstacle_coords = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstacle_coords = self.initial_obstacle_coords.copy()

        # Initialize pursuers
        for pursuer in self._pursuers:
            pursuer.set_position(self._generate_coord(pursuer.radius))
            pursuer.set_velocity(np.zeros(2))

        # Initialize evaders
        for evader in self._evaders:
            evader.set_position(self._generate_coord(evader.radius))
            evader.set_velocity(self._sample_velocity(evader.max_accel))

        # Initialize poisons
        for poison in self._poisons:
            poison.set_position(self._generate_coord(poison.radius))
            poison.set_velocity(self._sample_velocity(poison.max_accel))

        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_obs = self._collision_handling_impl()

    def _caught(
        self,
        is_colliding_x_y: np.ndarray,
        n_coop: int,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Check whether collision results in catching the object

        This is because you need `n_coop` agents to collide
        with the object to actually catch it.
        """
        # Number of collisions for each y
        n_collisions = is_colliding_x_y.sum(axis=0)
        # List of y that have been caught
        caught_y = np.where(n_collisions >= n_coop)[0]

        # Boolean array indicating which x caught any y in caught_y
        did_x_catch_y = is_colliding_x_y[:, caught_y]
        # List of x that caught corresponding y in caught_y
        x_caught_y = np.where(did_x_catch_y >= 1)[0]

        return caught_y, x_caught_y

    def _closest_dist(
        self,
        closest_object_idx: np.ndarray,
        input_sensorvals: np.ndarray,
    ) -> np.ndarray:
        """Closest distances according to `idx`"""
        sensorvals = []

        for pursuer_idx in range(self.n_pursuers):
            sensors = np.arange(self.n_sensors)  # sensor indices
            objects = closest_object_idx[pursuer_idx, ...]  # object indices
            sensorvals.append(input_sensorvals[pursuer_idx, ..., sensors, objects])

        return np.c_[sensorvals]

    def _extract_speed_features(
        self,
        object_velocities: np.ndarray,
        object_sensorvals: np.ndarray,
        sensed_mask: np.ndarray,
    ) -> np.ndarray:
        # sensed_mask is a boolean mask of which sensor values detected an object
        sensorvals = []
        for pursuer in self._pursuers:
            relative_speed = object_velocities - np.expand_dims(pursuer.velocity, 0)
            sensorvals.append(pursuer.sensors.dot(relative_speed.T))
        sensed_speed = np.c_[sensorvals]  # Speeds in direction of each sensor

        speed_features = np.zeros((self.n_pursuers, self.n_sensors))

        sensorvals = []
        for pursuer_idx in range(self.n_pursuers):
            sensorvals.append(
                sensed_speed[pursuer_idx, :, :][
                    np.arange(self.n_sensors), object_sensorvals[pursuer_idx, :]
                ]
            )
        # Set sensed values, all others remain 0
        speed_features[sensed_mask] = np.c_[sensorvals][sensed_mask]

        return speed_features

    def _collision_handling_impl(self) -> t.Tuple[t.List[np.ndarray]]:
        # Stop pursuers upon hitting a wall
        for pursuer in self._pursuers:
            clipped_coord = np.clip(pursuer.position, 0, 1)
            clipped_velocity = pursuer.velocity
            # If x or y position gets clipped, set x or y velocity to 0 respectively
            clipped_velocity[pursuer.position != clipped_coord] = 0
            # Save clipped velocity and position
            pursuer.set_velocity(clipped_velocity)
            pursuer.set_position(clipped_coord)

        def rebound_archeas(archeas: t.List[Archea]) -> None:
            collisions_archea_obstacle = np.zeros(len(archeas))
            # Archeas rebound on hitting an obstacle
            for idx, archea in enumerate(archeas):
                obstacle_distance = spd.cdist(
                    np.expand_dims(archea.position, 0), self.obstacle_coords
                )
                is_colliding = obstacle_distance <= archea.radius + self.obstacle_radius
                collisions_archea_obstacle[idx] = is_colliding.sum()
                if collisions_archea_obstacle[idx] > 0:
                    # Rebound the archea that collided with an obstacle
                    velocity_scale = (
                        archea.radius
                        + self.obstacle_radius
                        - spd.euclidean(archea.position, self.obstacle_coords)
                    )
                    pos_diff = archea.position - self.obstacle_coords[0]
                    new_pos = archea.position + velocity_scale * pos_diff
                    archea.set_position(new_pos)

                    collision_normal = archea.position - self.obstacle_coords[0]
                    # project current velocity onto collision normal
                    current_vel = archea.velocity
                    proj_numer = np.dot(current_vel, collision_normal)
                    cllsn_mag = np.dot(collision_normal, collision_normal)
                    proj_vel = (proj_numer / cllsn_mag) * collision_normal
                    perp_vel = current_vel - proj_vel
                    total_vel = perp_vel - proj_vel
                    archea.set_velocity(total_vel)

        rebound_archeas(self._pursuers)
        rebound_archeas(self._evaders)
        rebound_archeas(self._poisons)

        positions_pursuer = np.array([pursuer.position for pursuer in self._pursuers])
        positions_evader = np.array([evader.position for evader in self._evaders])
        positions_poison = np.array([poison.position for poison in self._poisons])

        # Find evader collisions
        distances_pursuer_evader = spd.cdist(positions_pursuer, positions_evader)
        pursuer_evader_threshold = self.pursuer_radius + self.evader_radius
        collisions_pursuer_evader = distances_pursuer_evader <= pursuer_evader_threshold

        # Number of collisions depends on n_coop, how many are needed to catch an evader
        caught_evaders, pursuer_evader_catches = self._caught(
            collisions_pursuer_evader, self.n_coop
        )

        # Find poison collisions
        distances_pursuer_poison = spd.cdist(positions_pursuer, positions_poison)
        pursuer_poison_threshold = self.pursuer_radius + self.poison_radius
        collisions_pursuer_poison = distances_pursuer_poison <= pursuer_poison_threshold

        caught_poisons, pursuer_poison_collisions = self._caught(
            collisions_pursuer_poison, 1
        )

        # Find sensed obstacles
        sensorvals_pursuer_obstacle = [
            pursuer.sensed(self.obstacle_coords, self.obstacle_radius)
            for pursuer in self._pursuers
        ]

        # Find sensed barriers
        sensorvals_pursuer_barrier = [
            pursuer.sense_barriers() for pursuer in self._pursuers
        ]

        # Find sensed evaders
        sensorvals_pursuer_evader = [
            pursuer.sensed(positions_evader, self.evader_radius)
            for pursuer in self._pursuers
        ]

        # Find sensed poisons
        sensorvals_pursuer_poison = [
            pursuer.sensed(positions_poison, self.poison_radius)
            for pursuer in self._pursuers
        ]

        # Find sensed pursuers
        sensorvals_pursuer_pursuer = [
            self._pursuers[idx].sensed(
                positions_pursuer, self.pursuer_radius, same_idx=idx
            )
            for idx in range(self.n_pursuers)
        ]

        # Collect distance features
        def sensor_features(
            sensorvals: t.List[np.ndarray],
        ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
            sensorvals = np.stack(sensorvals, axis=0)
            closest_idx_array = np.argmin(sensorvals, axis=2)
            closest_distances = self._closest_dist(closest_idx_array, sensorvals)
            finite_mask = np.isfinite(closest_distances)
            sensed_distances = np.ones((self.n_pursuers, self.n_sensors))
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

        # If object collided with required number of players,
        # reset its position and velocity
        # Effectively the same as removing it and adding it back
        def reset_caught_archeas(
            caught_archeas: np.ndarray,
            archeas: t.List[Archea],
        ) -> None:
            if len(caught_archeas) == 0:
                return

            for archea_idx in caught_archeas:
                archea = archeas[archea_idx]
                archea.set_position(self._generate_coord(archea.radius))
                # NOTE (kngwyu): Changed from the original code
                archea.set_velocity(self._sample_velocity(archea.max_accel))

        reset_caught_archeas(caught_evaders, self._evaders)
        reset_caught_archeas(caught_poisons, self._poisons)

        pursuer_evader_encounters, pursuer_evader_encounter_matrix = self._caught(
            collisions_pursuer_evader, 1
        )

        # TODO(kngwyu): How to use rewards?
        rewards = np.zeros(self.n_pursuers)
        rewards[pursuer_evader_catches] += self.food_reward
        rewards[pursuer_poison_collisions] += self.poison_reward
        rewards[pursuer_evader_encounter_matrix] += self.encounter_reward

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

            sensorfeatures = np.c_[
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
            sensorfeatures = np.c_[
                obstacle_distance_features,
                barrier_distance_features,
                evader_distance_features,
                poison_distance_features,
                pursuer_distance_features,
            ]

        return self._observation_list(
            sensorfeatures,
            collisions_pursuer_evader,
            collisions_pursuer_poison,
        )

    def _observation_list(
        self,
        sensor_feature: np.ndarray,
        is_colliding_evader: np.ndarray,
        is_colliding_poison: np.ndarray,
    ) -> t.List[np.ndarray]:
        obslist = []
        for pursuer_idx in range(self.n_pursuers):
            obslist.append(
                np.concatenate(
                    [
                        sensor_feature[pursuer_idx, ...].ravel(),
                        [
                            float((is_colliding_evader[pursuer_idx, :]).sum() > 0),
                            float((is_colliding_poison[pursuer_idx, :]).sum() > 0),
                        ],
                    ]
                )
            )
        return obslist

    def act(self, pursuer: Pursuer, action: np.ndarray) -> None:
        action = np.asarray(action)
        action = action.reshape(2)
        speed = np.linalg.norm(action)
        if speed > self.pursuer_max_accel:
            # Limit added thrust to self.pursuer_max_accel
            action = action / speed * self.pursuer_max_accel

        pursuer.set_velocity(p.velocity + action)
        pursuer.set_position(p.position + self.cycle_time * p.velocity)

        # Penalize large thrusts
        # TODO (kngwyu): how to use this?
        accel_penalty = self.thrust_penalty * np.sqrt((action ** 2).sum())

    def step(self) -> None:
        def move_archeas(archeas: t.List[Archea]) -> None:
            for archea in archeas:
                # Move archeaects
                archea.set_position(archea.position + self.cycle_time * archea.velocity)
                # Bounce archeaect if it hits a wall
                for i in range(len(archea.position)):
                    if archea.position[i] >= 1 or archea.position[i] <= 0:
                        archea.position[i] = np.clip(archea.position[i], 0, 1)
                        archea.velocity[i] = -1 * archea.velocity[i]

        move_archeas(self._evaders)
        move_archeas(self._poisons)

        self.last_obs = self._collision_handling_impl()

        # TODO (kngwyu): Rewards and events?

        self.frames += 1

    def observe(self, body: Body) -> np.ndarray:
        return np.array(self.last_obs[agent], dtype=np.float32)

    def render(self, mode: str = "human") -> t.Union[None, np.ndarray]:
        if mode not in ["human", "rgb-array"]:
            raise ValueError(f"Invalid mode: {mode}")

        if self._viewer is None:
            try:
                import pygame
            except ImportError as e:
                raise ImportError("Rendering waterworld needs pygame") from e

            if mode == "human":
                pygame.display.init()
                screen = pygame.display.set_mode((self.pixel_scale, self.pixel_scale))
            else:
                screen = pygame.Surface((self.pixel_scale, self.pixel_scale))
            self._viewer = _Viewer(screen, self.pixel_scale, pygame)

        self._viewer.draw_background()
        self._viewer.draw_obstacles(self.obstacle_coords, self.obstacle_radius)
        self._viewer.draw_archeas(self._pursuers, (101, 104, 209))
        self._viewer.draw_archeas(self._evaders, (238, 116, 106))
        self._viewer.draw_archeas(self._poisons, (145, 250, 116))

        if mode == "human":
            self._viewer.pygame.display.flip()
        else:
            observation = self._viewer.pygame.surfarray.pixels3d(self.screen)
            return np.transpose(observation.copy(), axes=(1, 0, 2))


class _Viewer:
    def __init__(self, screen: t.Any, pixel_scale: int, pygame: "module") -> None:
        self.pygame = pygame

        self._pixel_scale = pixel_scale
        self._screen = screen

    def close(self) -> None:
        self.pygame.display.quit()
        self.pygame.quit()

    def draw_archeas(
        self,
        archeas: t.List[Archea],
        color: t.Tuple[int, int, int],
    ) -> None:
        for archea in archeas:
            x, y = archea.position
            center = int(self._pixel_scale * x), int(self._pixel_scale * y)
            if isinstance(archea, Pursuer):
                for sensor in archea._sensors:
                    start = center
                    end = center + self._pixel_scale * (archea.sensor_range * sensor)
                    self.pygame.draw.line(self._screen, (0, 0, 0), start, end, 1)
            self.pygame.draw.circle(
                self._screen,
                color,
                center,
                self._pixel_scale * archea.radius,
            )

    def draw_background(self) -> None:
        # -1 is building pixel flag
        rect = self.pygame.Rect(0, 0, self._pixel_scale, self._pixel_scale)
        self.pygame.draw.rect(self._screen, (255, 255, 255), rect)

    def draw_obstacles(self, obstacle_coords: np.ndarray, radius: float) -> None:
        for obstacle in obstacle_coords:
            x, y = obstacle
            center = int(self._pixel_scale * x), int(self._pixel_scale * y)
            self.pygame.draw.circle(
                self._screen,
                (120, 176, 178),
                center,
                self._pixel_scale * radius,
            )


if __name__ == "__main__":
    env = WaterWorld(speed_features=False)
    env.reset()

    for i in range(100):
        for p in env._pursuers:
            env.act(p, p.action_space.sample())
        env.step()
        env.render()
    env.close()
