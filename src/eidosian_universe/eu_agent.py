"""Agent definition for the Eidosian Universe."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Agent:
    """Single agent within the Eidosian Universe."""

    x: float
    y: float
    dx: float
    dy: float
    energy: float
    color: Tuple[int, int, int]

    def update(self, config: "UniverseConfig") -> None:
        """Update agent position and energy state."""
        self.x += self.dx
        self.y += self.dy
        self.energy -= config.energy_decay

        # Bounce on boundaries
        if self.x < 0 or self.x > config.width:
            self.dx *= -1
        if self.y < 0 or self.y > config.height:
            self.dy *= -1

    def can_reproduce(self, config: "UniverseConfig") -> bool:
        """Return True if agent has enough energy to reproduce."""
        return self.energy >= config.reproduction_energy

    def reproduce(self, config: "UniverseConfig") -> "Agent":
        """Create a mutated child agent."""
        self.energy *= 0.5
        child_energy: float = self.energy
        return Agent(
            x=self.x + random.uniform(-10, 10),
            y=self.y + random.uniform(-10, 10),
            dx=random.uniform(-1, 1),
            dy=random.uniform(-1, 1),
            energy=child_energy,
            color=tuple(min(255, max(0, c + random.randint(-20, 20))) for c in self.color),
        )
