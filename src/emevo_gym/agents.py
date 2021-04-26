import dataclasses

from abc import ABC, abstractmethod
from typing import Type

import numpy as np


@dataclasses.dataclass()
class AgentConfig:
    initial_n_agents: int
    manager_class: Type[AgentManager]


def make_initial_agents(config: AgentConfig) -> AgentManager:
    pass


class AgentManager(ABC):
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.next_agent_id = 0

    @abstractmethod
    def create_new_agent(self) -> None:
        pass

    @abstractmethod
    def remove_dead_agents(self) -> None:
        pass


class Agent(ABC):
    agent_id: int
    is_dead: bool

    @abstractmethod
    def encode_gene(self) -> np.ndarray:
        pass

    @abstractmethod
    def learn(
        self,
        obs: Observation,
        action: np.ndarray,
        reward: float,
        next_obs: Observation,
    ) -> None:
        """Update policy/vf parameters from a previous experience."""
        pass

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select an action based on an observation."""
        pass
