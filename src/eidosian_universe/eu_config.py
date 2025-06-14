"""Configuration for the Eidosian Universe simulation."""

from dataclasses import dataclass

@dataclass
class UniverseConfig:
    """Settings controlling universe simulation parameters."""

    width: int = 800
    height: int = 600
    background_color: tuple[int, int, int] = (10, 10, 30)
    agent_count: int = 50
    max_agents: int = 200
    agent_size: int = 4
    energy_decay: float = 0.1
    reproduction_energy: float = 80.0
    starting_energy: float = 50.0
