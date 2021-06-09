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
import pygame
from scipy.spatial import distance as spd

from gym import spaces
from gym.utils import seeding


class Archea:
    def __init__(
        self,
        idx: int,  # TODO(kngwyu): replace this?
        *,
        radius: float,
        n_sensors: int,
        max_accel: float,
        sensor_range: float = 0.0,
        speed_features: bool = True,
    ) -> None:
        # Public members
        self.radius = radius
        self.sensor_range = sensor_range

        # Hidden members
        self._idx = idx
        self._n_sensors = n_sensors
        self._max_accel = max_accel
        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 5
        if speed_features:
            self._sensor_obscoord += 3
        self._sensor_obs_coord = self._n_sensors * self._sensor_obscoord
        # +1 for is_colliding_evader, +1 for is_colliding_poison
        self._obs_dim = self._sensor_obs_coord + 2

        self._position = None
        self._velocity = None

        # Generate self._n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it
        # because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self._n_sensors + 1)[:-1]
        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors

    @property
    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.float32(-np.sqrt(2)),
            high=np.float32(2 * np.sqrt(2)),
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.float32(-self._max_accel),
            high=np.float32(self._max_accel),
            shape=(2,),
            dtype=np.float32,
        )

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

    @property
    def sensors(self) -> np.ndarray:
        assert self._sensors is not None
        return self._sensors

    def sensed(
        self,
        object_coord: np.ndarray,
        object_radius: np.ndarray,
        same: bool = False,
    ) -> None:
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
        if same:
            # Set sensors values for sensing the current object to np.inf
            sensorvals[:, self._idx - 1] = np.inf
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
        evader_speed: float = 0.01,
        poison_speed: float = 0.01,
        poison_reward: float = -1.0,
        food_reward: float = 10.0,
        encounter_reward: float = 0.01,
        thrust_penalty: float = -0.5,
        local_ratio: float = 1.0,
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
        evador_radius_ratio: ratio of evador's radius to pursuer's radious
        poison_radius_ratio: ratio of poison's radius to pursuer's radious
        obstacle_radius: radius of obstacle object
        obstacle_coord: coordinate of obstacle object.
                        Can be set to `None` to use a random location
        pursuer_max_accel: pursuer archea maximum acceleration (maximum action size)
        evader_speed: evading archea speed
        poison_speed: poison archea speed
        poison_reward: reward for pursuer consuming a poison object (typically negative)
        food_reward: reward for pursuers consuming an evading archea
        encounter_reward: reward for a pursuer colliding with an evading archea
        thrust_penalty: scaling factor for the negative reward
                        used to penalize large actions
        local_ratio: Proportion of reward allocated locally
                     vs distributed globally among all agents
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
        self.evader_speed = evader_speed
        self.poison_speed = poison_speed
        self.pursuer_radius = pursuer_radius
        self.evader_radius = pursuer_radius * evader_radius_ratio
        self.poison_radius = pursuer_radius * poison_radius_ratio
        self.n_sensors = n_sensors
        self.sensor_range = np.ones(self.n_pursuers) * min(
            sensor_range, np.ceil(np.sqrt(2) * 100) / 100.0
        )
        self.poison_reward = poison_reward
        self.food_reward = food_reward
        self.thrust_penalty = thrust_penalty
        self.encounter_reward = encounter_reward
        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = [None for _ in range(self.n_pursuers)]

        self.n_obstacles = n_obstacles
        self.local_ratio = local_ratio
        self._speed_features = speed_features
        self.max_cycles = max_cycles
        self.seed()
        # TODO: Look into changing hardcoded radius ratios
        self._pursuers = [
            Archea(
                pursuer_idx + 1,
                radius=self.pursuer_radius,
                n_sensors=self.n_sensors,
                sensor_range=sensor_range,
                max_accel=self.pursuer_max_accel,
                speed_features=self._speed_features,
            )
            for pursuer_idx in range(self.n_pursuers)
        ]
        self._evaders = [
            Archea(
                evader_idx + 1,
                radius=self.evader_radius,
                n_sensors=self.n_pursuers,
                max_accel=self.evader_speed,
            )
            for evader_idx in range(self.n_evaders)
        ]
        self._poisons = [
            Archea(
                poison_idx + 1,
                radius=self.poison_radius,
                n_sensors=self.n_poison,
                max_accel=self.poison_speed,
            )
            for poison_idx in range(self.n_poison)
        ]

        self.num_agents = self.n_pursuers
        self.action_space = [agent.action_space for agent in self._pursuers]
        self.observation_space = [agent.observation_space for agent in self._pursuers]

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

    def reset(self) -> np.ndarray:
        """
        Reset the state and returns the observation of the first agent.
        TODO: return value?
        """

        self.frames = 0
        # Initialize obstacles
        if self.initial_obstacle_coords is None:
            # Generate obstacle positions in range [0, 1)
            self.obstacle_coords = self.np_random.rand(self.n_obstacles, 2)
        else:
            self.obstacle_coords = self.initial_obstacle_coords.copy()
        # Set each obstacle's velocity to 0
        # TODO: remove if obstacles should never move
        self.obstacle_speeds = np.zeros((self.n_obstacles, 2))

        # Initialize pursuers
        for pursuer in self._pursuers:
            pursuer.set_position(self._generate_coord(pursuer.radius))
            pursuer.set_velocity(np.zeros(2))

        # Initialize evaders
        for evader in self._evaders:
            evader.set_position(self._generate_coord(evader.radius))
            # Generate velocity such that speed <= self.evader_speed
            velocity = self.np_random.rand(2) - 0.5
            speed = np.linalg.norm(velocity)
            if speed > self.evader_speed:
                # Limit speed to self.evader_speed
                velocity = velocity / speed * self.evader_speed
            evader.set_velocity(velocity)

        # Initialize poisons
        for poison in self._poisons:
            poison.set_position(self._generate_coord(poison.radius))
            # Generate both velocity components
            # from range [-self.poison_speed, self.poison_speed)
            # Generate velocity such that speed <= self.poison_speed
            velocity = self.np_random.rand(2) - 0.5
            speed = np.linalg.norm(velocity)
            if speed > self.poison_speed:
                # Limit speed to self.poison_speed
                velocity = velocity / speed * self.poison_speed
            poison.set_velocity(velocity)

        rewards = np.zeros(self.n_pursuers)
        (
            sensor_features,
            collided_pursuer_evader,
            collided_pursuer_poison,
            rewards,
        ) = self._collision_handling_impl(rewards)
        obs_list = self.observe_list(
            sensor_features, collided_pursuer_evader, collided_pursuer_poison
        )
        self.last_rewards = [np.float64(0) for _ in range(self.n_pursuers)]
        self.control_rewards = [0 for _ in range(self.n_pursuers)]
        self.last_dones = [False for _ in range(self.n_pursuers)]
        self.last_obs = obs_list

        return obs_list[0]

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

    def _collision_handling_impl(
        self,
        rewards: np.ndarray,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Stop pursuers upon hitting a wall
        for pursuer in self._pursuers:
            clipped_coord = np.clip(pursuer.position, 0, 1)
            clipped_velocity = pursuer.velocity
            # If x or y position gets clipped, set x or y velocity to 0 respectively
            clipped_velocity[pursuer.position != clipped_coord] = 0
            # Save clipped velocity and position
            pursuer.set_velocity(clipped_velocity)
            pursuer.set_position(clipped_coord)

        def rebound_particles(particles, n):
            collisions_particle_obstacle = np.zeros(n)
            # Particles rebound on hitting an obstacle
            for idx, particle in enumerate(particles):
                obstacle_distance = spd.cdist(
                    np.expand_dims(particle.position, 0), self.obstacle_coords
                )
                is_colliding = (
                    obstacle_distance <= particle.radius + self.obstacle_radius
                )
                collisions_particle_obstacle[idx] = is_colliding.sum()
                if collisions_particle_obstacle[idx] > 0:
                    # Rebound the particle that collided with an obstacle
                    velocity_scale = (
                        particle.radius
                        + self.obstacle_radius
                        - spd.euclidean(particle.position, self.obstacle_coords)
                    )
                    pos_diff = particle.position - self.obstacle_coords[0]
                    new_pos = particle.position + velocity_scale * pos_diff
                    particle.set_position(new_pos)

                    collision_normal = particle.position - self.obstacle_coords[0]
                    # project current velocity onto collision normal
                    current_vel = particle.velocity
                    proj_numer = np.dot(current_vel, collision_normal)
                    cllsn_mag = np.dot(collision_normal, collision_normal)
                    proj_vel = (proj_numer / cllsn_mag) * collision_normal
                    perp_vel = current_vel - proj_vel
                    total_vel = perp_vel - proj_vel
                    particle.set_velocity(total_vel)

        rebound_particles(self._pursuers, self.n_pursuers)

        rebound_particles(self._evaders, self.n_evaders)
        rebound_particles(self._poisons, self.n_poison)

        positions_pursuer = np.array([pursuer.position for pursuer in self._pursuers])
        positions_evader = np.array([evader.position for evader in self._evaders])
        positions_poison = np.array([poison.position for poison in self._poisons])

        # Find evader collisions
        distances_pursuer_evader = spd.cdist(positions_pursuer, positions_evader)
        # Generate n_evaders x n_pursuers matrix of boolean values for collisions
        collisions_pursuer_evader = distances_pursuer_evader <= np.asarray(
            [
                pursuer.radius + evader.radius
                for pursuer in self._pursuers
                for evader in self._evaders
            ]
        ).reshape(self.n_pursuers, self.n_evaders)

        # Number of collisions depends on n_coop, how many are needed to catch an evader
        caught_evaders, pursuer_evader_catches = self._caught(
            collisions_pursuer_evader, self.n_coop
        )

        # Find poison collisions
        distances_pursuer_poison = spd.cdist(positions_pursuer, positions_poison)
        collisions_pursuer_poison = distances_pursuer_poison <= np.asarray(
            [
                pursuer.radius + poison.radius
                for pursuer in self._pursuers
                for poison in self._poisons
            ]
        ).reshape(self.n_pursuers, self.n_poison)

        caught_poisons, pursuer_poison_collisions = self._caught(
            collisions_pursuer_poison, 1
        )

        # Find sensed obstacles
        sensorvals_pursuer_obstacle = np.array(
            [
                pursuer.sensed(self.obstacle_coords, self.obstacle_radius)
                for pursuer in self._pursuers
            ]
        )

        # Find sensed barriers
        sensorvals_pursuer_barrier = np.array(
            [pursuer.sense_barriers() for pursuer in self._pursuers]
        )

        # Find sensed evaders
        sensorvals_pursuer_evader = np.array(
            [
                pursuer.sensed(positions_evader, self.evader_radius)
                for pursuer in self._pursuers
            ]
        )

        # Find sensed poisons
        sensorvals_pursuer_poison = np.array(
            [
                pursuer.sensed(positions_poison, self.poison_radius)
                for pursuer in self._pursuers
            ]
        )

        # Find sensed pursuers
        sensorvals_pursuer_pursuer = np.array(
            [
                pursuer.sensed(positions_pursuer, self.pursuer_radius, same=True)
                for pursuer in self._pursuers
            ]
        )

        # Collect distance features
        def sensor_features(sensorvals):
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

        # Collect speed features
        pursuers_speed = np.array([pursuer.velocity for pursuer in self._pursuers])
        evaders_speed = np.array([evader.velocity for evader in self._evaders])
        poisons_speed = np.array([poison.velocity for poison in self._poisons])

        evader_speed_features = self._extract_speed_features(
            evaders_speed, closest_evader_idx, evader_mask
        )
        poison_speed_features = self._extract_speed_features(
            poisons_speed, closest_poison_idx, poison_mask
        )
        pursuer_speed_features = self._extract_speed_features(
            pursuers_speed, closest_pursuer_idx, pursuer_mask
        )

        # If object collided with required number of players,
        # reset its position and velocity
        # Effectively the same as removing it and adding it back
        def reset_caught_objects(caught_objects, objects, speed):
            if caught_objects.size:
                for object_idx in caught_objects:
                    objects[object_idx].set_position(
                        self._generate_coord(objects[object_idx].radius)
                    )
                    # Generate both velocity components from range
                    # [-self.evader_speed, self.evader_speed)
                    objects[object_idx].set_velocity(
                        (self.np_random.rand(2) - 0.5) * 2 * speed
                    )

        reset_caught_objects(caught_evaders, self._evaders, self.evader_speed)
        reset_caught_objects(caught_poisons, self._poisons, self.poison_speed)

        pursuer_evader_encounters, pursuer_evader_encounter_matrix = self._caught(
            collisions_pursuer_evader, 1
        )

        # Update reward based on these collisions
        rewards[pursuer_evader_catches] += self.food_reward
        rewards[pursuer_poison_collisions] += self.poison_reward
        rewards[pursuer_evader_encounter_matrix] += self.encounter_reward

        # Add features together
        if self._speed_features:
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

        return (
            sensorfeatures,
            collisions_pursuer_evader,
            collisions_pursuer_poison,
            rewards,
        )

    def observe_list(
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

    def step(self, action, agent_id):
        action = np.asarray(action)
        action = action.reshape(2)
        speed = np.linalg.norm(action)
        if speed > self.pursuer_max_accel:
            # Limit added thrust to self.pursuer_max_accel
            action = action / speed * self.pursuer_max_accel

        p = self._pursuers[agent_id]
        p.set_velocity(p.velocity + action)
        p.set_position(p.position + self.cycle_time * p.velocity)

        # Penalize large thrusts
        accel_penalty = self.thrust_penalty * np.sqrt((action ** 2).sum())
        # Average thrust penalty among all agents,
        # and assign each agent global portion designated by (1 - local_ratio)
        self.control_rewards = (
            (accel_penalty / self.n_pursuers)
            * np.ones(self.n_pursuers)
            * (1 - self.local_ratio)
        )
        # Assign the current agent the local portion designated by local_ratio
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio

        return self.observe(agent_id)

    def execute_pending_actions(self) -> None:
        def move_objects(objects):
            for obj in objects:
                # Move objects
                obj.set_position(obj.position + self.cycle_time * obj.velocity)
                # Bounce object if it hits a wall
                for i in range(len(obj.position)):
                    if obj.position[i] >= 1 or obj.position[i] <= 0:
                        obj.position[i] = np.clip(obj.position[i], 0, 1)
                        obj.velocity[i] = -1 * obj.velocity[i]

        move_objects(self._evaders)
        move_objects(self._poisons)

        rewards = np.zeros(self.n_pursuers)
        (
            sensorfeatures,
            collisions_pursuer_evader,
            collisions_pursuer_poison,
            rewards,
        ) = self._collision_handling_impl(rewards)
        obs_list = self.observe_list(
            sensorfeatures, collisions_pursuer_evader, collisions_pursuer_poison
        )
        self.last_obs = obs_list

        local_reward = rewards
        global_reward = local_reward.mean()
        # Distribute local and global rewards according to local_ratio
        self.last_rewards = local_reward * self.local_ratio + global_reward * (
            1 - self.local_ratio
        )

        self.frames += 1

    def observe(self, agent):
        return np.array(self.last_obs[agent], dtype=np.float32)

    def render(self, mode: str = "human") -> t.Union[None, np.ndarray]:
        if mode not in ["human", "rgb-array"]:
            raise ValueError(f"Invalid mode: {mode}")

        if self._viewer is None:
            if mode == "human":
                pygame.display.init()
                screen = pygame.display.set_mode((self.pixel_scale, self.pixel_scale))
            else:
                screen = pygame.Surface((self.pixel_scale, self.pixel_scale))
            self._viewer = _Viewer(screen, self.pixel_scale)

        self._viewer.draw_background()
        self._viewer.draw_obstacles(self.obstacle_coords, self.obstacle_radius)
        self._viewer.draw_archeas(self._pursuers, (101, 104, 209))
        self._viewer.draw_archeas(self._evaders, (238, 116, 106))
        self._viewer.draw_archeas(self._poisons, (145, 250, 116))

        if mode == "human":
            pygame.display.flip()
        else:
            observation = pygame.surfarray.pixels3d(self.screen)
            return np.transpose(observation.copy(), axes=(1, 0, 2))


class _Viewer:
    def __init__(self, screen: t.Any, pixel_scale: int) -> None:
        self._pixel_scale = pixel_scale
        self._screen = screen

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def draw_archeas(
        self,
        archeas: t.List[Archea],
        color: t.Tuple[int, int, int],
    ) -> None:
        for archea in archeas:
            x, y = archea.position
            center = int(self._pixel_scale * x), int(self._pixel_scale * y)
            if archea.sensor_range > 0.01:
                for sensor in archea._sensors:
                    start = center
                    end = center + self._pixel_scale * (archea.sensor_range * sensor)
                    pygame.draw.line(self._screen, (0, 0, 0), start, end, 1)
            pygame.draw.circle(
                self._screen, color, center, self._pixel_scale * archea.radius
            )

    def draw_background(self) -> None:
        # -1 is building pixel flag
        rect = pygame.Rect(0, 0, self._pixel_scale, self._pixel_scale)
        pygame.draw.rect(self._screen, (255, 255, 255), rect)

    def draw_obstacles(self, obstacle_coords: np.ndarray, radius: float) -> None:
        for obstacle in obstacle_coords:
            x, y = obstacle
            center = int(self._pixel_scale * x), int(self._pixel_scale * y)
            pygame.draw.circle(
                self._screen,
                (120, 176, 178),
                center,
                self._pixel_scale * radius,
            )


if __name__ == "__main__":
    env = WaterWorld()
    env.reset()

    def policy():
        return env.action_space[0].sample()

    for i in range(100):
        for agent in env._pursuers:
            act = policy()
            env.step(act, agent._idx - 1)
        env.execute_pending_actions()
        env.render()
    env.close()
