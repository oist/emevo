"""Simulate random agents with birth and death
"""

import dataclasses
import typing as t

from functools import partial

import numpy as np

from emevo import Body, Environment, birth_and_death as bd, make


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
    energy_update_scale: float = 1.0,
    asexual: bool = False,
    render: bool = False,
) -> None:
    environment.reset()

    def energy_update(*, food: int, poison: int, others: int, energy: float) -> float:
        return (food - poison - energy) * energy_update_scale

    # Initialize agents
    agents = {}
    for body in environment.available_bodies():
        manager.register(body)
        agent = Agent(body)
        agent.previous_observation, _ = environment.observe(body)
        agents[body] = agent

    repr_energy_update = [-3.0, -6.0][int(asexual)] * energy_update_scale

    # Enviromental loop
    for _ in range(max_steps):

        # Act
        for body, agent in agents.items():
            action = agent.select_action()
            environment.act(body, action)
            agent.previous_action = action

        # Step
        encounts = environment.step()

        # Observe
        for body, agent in agents.items():
            obs, info = environment.observe(body)
            agent.previous_observation = obs
            manager.update_status(body, energy_update=energy_update(**info))

        # If the mating succeeds, parents consume some energy
        if manager.is_asexual:
            for body in agents.keys():
                if manager.reproduce(body) is not None:
                    manager.update_status(body, energy_update=repr_energy_update)
        else:
            for encount in encounts:
                if manager.reproduce(encount) is not None:
                    for body in encount.bodies:
                        manager.update_status(body, energy_update=repr_energy_update)

        deads, newborns = manager.step()

        for dead in deads:
            print(f"{dead.body} is dead with {dead.status}")
            environment.die(dead.body)
            del agents[dead.body]

        for newborn in newborns:
            body = environment.born(
                newborn.context.generation + 1,
                newborn.context.position,
            )
            agents[body] = Agent(body)
            manager.register(body)

        if render:
            environment.render()


def main() -> None:
    import argparse

    from emevo.environments import waterworld as ww

    def create_parser(*args) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for args_i, kwargs_i in args:
            if isinstance(args_i, tuple):
                parser.add_argument(*args_i, **kwargs_i)
            else:
                parser.add_argument(args_i, **kwargs_i)

        return parser

    parser = create_parser(
        ("max_steps", dict(type=int, help="Max enviromental steps to simulate")),
        ("--no-render", dict(action="store_true", help="Disable rendering by pygame")),
        ("--asexual", dict(action="store_true", help="Use asexual mating")),
        ("--n-evaders", dict(type=int, default=10, help="Initial number of evaders")),
        ("--n-poison", dict(type=int, default=14, help="Initial number of poison")),
        (
            "--growth-rate",
            dict(type=float, default=1.0, help="Growth rate of food and poison"),
        ),
        (
            ("-CR", "--constrained-repr"),
            dict(action="store_true", help="Use bacteria_constarained_repr"),
        ),
        (
            ("-GA", "--gompertz-alpha"),
            dict(type=float, default=0.001, help="α for Gompertz hazard function"),
        ),
        (
            ("-ES", "--energy-scale"),
            dict(type=float, default=1.0, help="Scaling for energy update"),
        ),
        (
            ("-GE", "--gompertz-energy"),
            dict(
                type=int,
                default=10,
                help="Energy normalization function for Gompertz hazard function",
            ),
        ),
    )

    args = parser.parse_args()
    env = make(
        "Waterworld-v0",
        n_evaders=args.n_evaders,
        n_poison=args.n_poison,
        **ww.repr_functions(
            kind="constrained" if args.constrained_repr else "logistic",
            n_evaders=args.n_evaders,
            n_poison=args.n_poison,
            growth_rate=args.growth_rate,
        ),
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

    ge = args.gompertz_energy
    manager = bd.Manager(
        status_fn=partial(
            bd.statuses.AgeAndEnergy,
            age=1,
            energy=0.0,
            energy_delta=0.001,
        ),
        death_prob_fn=bd.death_functions.gompertz_hazard(
            energy_threshold=-ge,
            energy_min=-ge,
            energy_max=ge,
            gompertz_alpha=args.gompertz_alpha,
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


if __name__ == "__main__":
    main()
