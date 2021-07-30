"""Simulate random agents with birth and death
"""

import dataclasses
import operator
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


def env_loop(
    *,
    environment: Environment,
    manager: bd.Manager,
    max_steps: int,
    asexual: bool = False,
    render: bool = False,
) -> None:
    environment.reset()

    def energy_update(food: int, poison: int, energy: float) -> float:
        return food - poison - energy

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
            manager.update_status(agent.body, energy_update=energy_update(**info))

        # If the mating succeeds, parents consume some energy
        if asexual:
            for body in map(operator.attrgetter("body"), agents):
                if manager.reproduce(body):
                    manager.update_status(body, energy_update=-6.0)
        else:
            for encount in encounts:
                if manager.reproduce(encount):
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
    from emevo.environments import waterworld as ww

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_steps",
        type=int,
        help="Max enviromental steps to simulate",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering by pygame",
    )
    parser.add_argument(
        "--asexual",
        action="store_true",
        help="Use asexual mating",
    )
    parser.add_argument(
        "-CR",
        "--constrained-repr",
        action="store_true",
        help="Use asexual mating",
    )
    args = parser.parse_args()
    if args.constrained_repr:
        env = make(
            "Waterworld-v0",
            n_evaders=10,
            n_poison=16,
            evader_reproduce_fn=ww.bacteria_constrained_repr(20, 1.0, 20),
            poison_reproduce_fn=ww.bacteria_constrained_repr(30, 1.0, 30),
        )
    else:
        env = make(
            "Waterworld-v0",
            n_evaders=10,
            n_poison=16,
            evader_reproduce_fn=ww.logistic_repr(0.8, 14),
            poison_reproduce_fn=ww.logistic_repr(0.8, 16),
        )

    if args.asexual:
        repr_manager = bd.AsexualReprManager(
            success_prob=bd.repr_functions.log(2.0, 0.1),
            produce=lambda status, body: bd.Oviparous(
                context=GeneticContext(body.profile.generation, body.position),
                time_to_birth=3,
            ),
        )
    else:
        repr_manager = bd.SexualReprManager(
            success_prob=bd.repr_functions.log_prod(4.0, 0.1),
            produce=lambda statuses, encount: bd.Oviparous(
                context=GeneticContext(
                    encount.bodies[0].profile.generation,
                    encount.bodies[0].position,
                ),
                time_to_birth=3,
            ),
        )

    manager = bd.Manager(
        default_status_fn=partial(
            bd.statuses.AgeAndEnergy,
            age=1,
            energy=0.0,
            energy_delta=0.001,
        ),
        death_prob_fn=bd.death_functions.gompertz_hazard(
            energy_threshold=-10.0,
            energy_to_gompertz_r=bd.death_functions.energy_to_gompertz_r(-10.0, 10.0),
            gompertz_alpha=0.001,
        ),
        repr_manager=repr_manager,
    )

    env_loop(
        environment=env,
        manager=manager,
        max_steps=args.max_steps,
        asexual=args.asexual,
        render=not args.no_render,
    )
