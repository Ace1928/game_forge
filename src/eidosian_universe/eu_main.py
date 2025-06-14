"""Eidosian Universe: a simple emergent simulation."""

from __future__ import annotations

import sys
import pygame

from .eu_config import UniverseConfig
from .eu_world import Universe


def main() -> None:
    """Run the Eidosian Universe simulation."""
    pygame.init()
    config = UniverseConfig()
    screen = pygame.display.set_mode((config.width, config.height))
    clock = pygame.time.Clock()
    universe = Universe(config)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                universe.agents.append(
                    universe._create_random_agent()
                )

        universe.update()
        universe.render(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
