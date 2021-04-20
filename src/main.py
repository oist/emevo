import dataclasses

from typing import Optional

import numpy as np

from config import Config
from environments import make_environment


def make_initial_agents(config: Config):
    pass


@dataclasses.dataclass()
class ObsAndAction:
    observation: np.ndarray
    action: Optional[np.ndarray] = None


def main_loop() -> None:
    config = Config()

    # Create agents and an environment based on the configuration
    agent_manager = make_initial_agents(config)
    environment = make_environment(config)
    environment.reset()

    # Each agent observes the initial state
    previous_obs_and_actions = {}
    for agent in agent_manager.available_agents():
        initial_obs = environment.assign_agent_randomly(agent.agent_id)
        previous_obs_and_actions[agent.agent_id] = ObsAndAction(initial_obs)

    for _ in range(config.max_environmental_steps):
        # Each agent acts in the environment
        for agent in agent_manager.available_agents():
            prev_obs = previous_obs_and_actions[agent.agent_id].observation
            action = agent.select_action(prev_obs)
            if action is None:  # This agent is dead!
                del previous_obs_and_actions[agent.agent_id]
                continue
            environment.append_pending_action(agent.agent_id, action)
            previous_obs_and_actions[agent.agent_id].action = action
        agent_manager.remove_dead_agents()
        # If an agent survives, then he/she learns from the previous experience
        for agent in agent_manager.available_agents():
            prev_obs, action = dataclasses.astuple(
                previous_obs_and_actions[agent.agent_id]
            )
            observation, reward = environment.get_observation(agent.agent_id)
            agent.learn(prev_obs, action, observation, reward)
            previous_obs_and_actions[agent.agent_id] = ObsAndAction(observation)
        # Create new agents if there are some suceessful matings
        successful_matings = environment.execute_pending_actions()
        for mating in successful_matings:
            agent = agent_manager.create_new_agent(mating.gene)
            initial_obs = environment.assign_agent_based_on_position(mating.position)
            previous_obs_and_actions[agent.agent_iid] = ObsAndAction(observation)

        # Some logging and visualization stuffs? I'm not sure now.


def main() -> None:
    pass


if __name__ == "__main___":
    main()
