"""Simulate random agents with birth and death
"""

import dataclasses
import typing as t
from functools import partial

import numpy as np

from emevo import Body, Environment, make
from emevo import birth_and_death as bd


@dataclasses.dataclass()
class Agent:
    """A simple agent that learns nothing"""

    body: Body
    previous_observation: t.Optional[np.ndarray] = None
    previous_action: t.Optional[np.ndarray] = None

    def select_action(self) -> np.ndarray:
        return self.body.action_space.sample()


@dataclasses.dataclass(frozen=True)
class GeneticContext:
    generation: int
    position: np.ndarray


def env_loop(environment: Environment, max_steps: int, render: bool = False) -> None:
    environment.reset()

    def energy_update(info: t.Dict[str, float]) -> float:
        return float(info["food"] - info["poison"])

    manager = bd.Manager(
        default_status_fn=partial(
            bd.AgeAndEnergy,
            age=1,
            energy=0.0,
            energy_delta=0.001,
        ),
        # death_prob_fn=bd.death_functions.hunger_or_infirmity(0.1, 1000.0),
        death_prob_fn=bd.death_functions.gompertz_hazard(
            energy_threshold=-10.0,
            energy_to_gompertz_r=bd.death_functions.energy_to_gompertz_r(-10.0, 10.0),
            gompertz_alpha=0.001,
        ),
        repr_manager=bd.SexualReprManager(
            success_prob=bd.repr_functions.logprod_success_prob(4.0, 0.1),
            produce=lambda statuses, encount: bd.Oviparous(
                context=GeneticContext(
                    encount.bodies[0].profile.generation,
                    encount.bodies[0].position,
                ),
                time_to_birth=3,
            ),
        ),
    )

    # Initialize agents
    agents = []
    for body in environment.available_bodies():
        agent = Agent(body)
        agents.append(agent)
        manager.register(body)
        obs, _ = environment.observe(body)
        agent.previous_observation = obs

    # Do some experiments
    for _ in range(max_steps):

        # Each agent acts in the environment
        for agent in agents:
            action = agent.select_action()
            environment.act(agent.body, action)
            agent.previous_action = action

        # Step the simulator
        encounts = environment.step()

        # Collect information of each agents, and Update the status
        for agent in agents:
            _, info = environment.observe(agent.body)
            manager.update_status(agent.body, energy_update=energy_update(info))

        # If the mating succeeds, parents consume some energy
        for encount in encounts:
            if manager.reproduction(encount):
                for body in encount.bodies:
                    manager.update_status(body, energy_update=-3.0)

        deads, newborns = manager.step()

        for dead in deads:
            print(f"{dead.body} is dead with {dead.status}")
            environment.die(dead.body)
            remove_idx, _ = next(
                filter(lambda agent: agent[1].body == dead.body, enumerate(agents))
            )
            agents.pop(remove_idx)

        for newborn in newborns:
            body = environment.born(
                newborn.context.generation,
                newborn.context.position,
            )
            agents.append(Agent(body))
            manager.register(body)

        if render:
            environment.render()


if __name__ == "__main__":
    import argparse
    from emevo.environments.waterworld import logistic_reproduce_fn

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_steps",
        type=int,
        help="Max enviromental steps to simulate",
    )
    parser.add_argument(
        "--no-render",
        type=bool,
        default=False,
        help="Disable rendering by pygame",
    )
    args = parser.parse_args()
    env = make(
        "Waterworld-v0",
        n_evaders=10,
        n_poison=16,
        evader_reproduce_fn=logistic_reproduce_fn(1.0, 14),
        poison_reproduce_fn=logistic_reproduce_fn(1.0, 16),
    )
    env_loop(env, args.max_steps, render=not args.no_render)
