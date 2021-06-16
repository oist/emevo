import numpy as np

from emevo import Environment


class Agent:
    def __init__(self, body: Body) -> None:
        self.body = body
        self.previous_observation = None
        self.previous_action = None

    def select_action(self) -> np.ndarray:
        return self.body.action_space.sample()


def env_loop(environment: Environment, max_steps: int) -> None:
    environment.reset()

    # Initialize agents
    agents = []
    for body in environment.available_bodies():
        agent = Agent(body)
        agents.append(agent)
        obs, _ = environment.observe()
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

        # Each agent observe the state. Then it dies or learns from the experience.
        for agent in agents:
            obs, reward = environment.observe(agent.body)
            # Learning and logging

        for dead_agent_id in agent_manager.remove_dead_agents():
            del previous_observations[dead_agent_id]

        for child in children:
            agent = agent_manager.create_new_agent(child.gene)
            initial_obs = environment.place_agent(
                agent,
                mating.positional_info,
            )
            previous_observations[agent.agend_id] = initial_obs

        # Some logging and visualization stuffs?


if __name__ == "__main__":

    env_loop()
