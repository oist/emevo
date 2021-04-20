import dataclasses


@dataclasses.dataclass()
class EnsembleAgentConfig:
    initial_n_agents: int


@dataclasses.dataclass()
class Config:
    max_environmental_steps: int
    random_seed: int
