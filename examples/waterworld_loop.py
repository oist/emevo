"""Simulate random agents without birth and death
"""

import numpy as np

from emevo import Body, Environment, make


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

    # Initialize agents
    agents = []
    for body in environment.available_bodies():
        agent = Agent(body)
        agents.append(agent)
        obs, _ = environment.observe(body)
        agent.previous_observation = obs

    # Enviromental loop
    for _ in range(max_steps):

        # Act
        for agent in agents:
            action = agent.select_action()
            environment.act(agent.body, action)
            agent.previous_action = action

        # Step
        _ = environment.step()

        # Observe
        for agent in agents:
            _, info = environment.observe(agent.body)

        if render:
            environment.render()


def main() -> None:
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
    env = make("Waterworld-v0", n_evaders=8, n_poison=12)
    env_loop(env, args.max_steps, render=not args.no_render)


if __name__ == "__main__":
    main()
