import typing as t

import numpy as np

from emevo import Body, Encount, Environment, make
from emevo import birth_and_death as bd


class Agent:
    """A simple agent that learns nothing"""

    def __init__(self, body: Body) -> None:
        self.body = body
        self.previous_observation = None
        self.previous_action = None

    def select_action(self) -> np.ndarray:
        return self.body.action_space.sample()


def env_loop(environment: Environment, max_steps: int, render: bool = False) -> None:
    environment.reset()

    def repr_fn(
        statuses: t.Tuple[bd.Status, bd.Status],
        encount: Encount,
    ) -> t.Optional[bd.Oviparous]:
        ENERGY_THRESHOLD = 4.5

        if all(map(lambda s: s.energy_level > ENERGY_THRESHOLD, statuses)):
            parent = encount.bodies[0]
            return bd.Oviparous(
                context={
                    "position": parent.position,
                    "generation": parent.profile.generation,
                },
                time_to_birth=3,
            )
        else:
            return None

    def energy_update(info: t.Dict[str, float]) -> float:
        return info["food"] - info["poison"]

    manager = bd.Manager(
        default_status=bd.Status(3.0),
        is_dead=lambda status: status.energy_level < 1.5,
        sexual_repr_fn=repr_fn,
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
            manager.update(agent.body, energy_level=energy_update(info))

        # If the mating succeeds, parents consume some energy
        for encount in encounts:
            if manager.sexual_repr(encount):
                for body in encount.bodies:
                    manager.update(body, energy_level=-3.0)

        deads, newborns = manager.step()

        for dead in deads:
            # print(f"{dead} is dead at {dead.body.position}")
            environment.die(dead.body)
            remove_idx, _ = next(
                filter(lambda agent: agent[1].body == dead.body, enumerate(agents))
            )
            agents.pop(remove_idx)

        for newborn in newborns:
            body = environment.born(
                newborn.context["generation"], newborn.context["position"]
            )
            agents.append(Agent(body))
            manager.register(body)

        if render:
            environment.render()


if __name__ == "__main__":
    import argparse

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
    env = make("Waterworld-v0", n_poison=20)
    env_loop(env, args.max_steps, render=not args.no_render)
