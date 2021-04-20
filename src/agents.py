from abc import ABC, abstractmethod

from config import Config


class AgentManager:
    def __init__(self, config: Config) -> None:
        self.next_agent_id = 0


class Agent(ABC):
    agent_id: int

    @abstractmethod
    def learn(self) -> None:
        pass

    @abstractmethod
    def select_action(self):
        pass
