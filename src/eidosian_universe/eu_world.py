"""Core simulation logic for the Eidosian Universe."""

from __future__ import annotations

import random
from typing import List

import pygame

from .eu_agent import Agent
from .eu_config import UniverseConfig


class Universe:
    """Simulates a universe populated by agents."""

    def __init__(self, config: UniverseConfig) -> None:
        self.config = config
        self.agents: List[Agent] = []
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        """Create initial population of agents."""
        for _ in range(self.config.agent_count):
            self.agents.append(self._create_random_agent())

    def _create_random_agent(self) -> Agent:
        """Generate a random agent."""
        return Agent(
            x=random.uniform(0, self.config.width),
            y=random.uniform(0, self.config.height),
            dx=random.uniform(-1, 1),
            dy=random.uniform(-1, 1),
            energy=self.config.starting_energy,
            color=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
        )

    def update(self) -> None:
        """Advance simulation state."""
        new_agents: List[Agent] = []
        for agent in list(self.agents):
            agent.update(self.config)
            if agent.can_reproduce(self.config) and len(self.agents) < self.config.max_agents:
                new_agents.append(agent.reproduce(self.config))
            if agent.energy <= 0:
                self.agents.remove(agent)
        self.agents.extend(new_agents)

    def render(self, surface: pygame.Surface) -> None:
        """Draw all agents to the surface."""
        surface.fill(self.config.background_color)
        for agent in self.agents:
            pygame.draw.circle(
                surface,
                agent.color,
                (int(agent.x), int(agent.y)),
                self.config.agent_size,
            )
