from emevo.agent import AgentManager
from emevo.environment import Environment


class Agent:
    def __init__(self, body: Body) -> None:
        self.body = body


def env_loop(environment: Environment, max_steps: int) -> None:
    environment.reset()

    # Each agent observes the initial state
    previous_observations = WeakKeyDictionary()

    agents = []
    for body in environment.available_bodies():
        agents.append(Agent(body))
        observation = environment.observe(body)
        previous_observations[body] = observation

    for _ in range(max_steps):
        taken_actions = {}

        # Each agent acts in the environment
        for agent in agent_manager.available_agents():
            prev_obs = previous_observations[agent.agent_id]
            action = agent.select_action(prev_obs)
            environment.append_pending_action(agent.agent_id, action)
            taken_actions[agent.agend_id] = action

        # Execute pending actions
        children = environment.execute_pending_actions()

        # Each agent observe the state. Then it dies or learns from the experience.
        for agent in agent_manager.available_agents():
            prev_obs = previous_observations[agent.agent_id]
            obs = environment.observed_by(agent)
            # TODO: Do logging here?
            agent.observe(prev_obs, action, obs)
            previous_observations[agent.agend_id] = obs

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
