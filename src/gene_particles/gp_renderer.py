from typing import Dict, Tuple

import numpy as np
import pygame
from gp_config import SimulationConfig
from gp_types import CellularTypeData

###############################################################
# Renderer Class
###############################################################


class Renderer:
    """
    Handles rendering of cellular components on a Pygame surface with optimized visualization.

    Efficiently manages drawing of particles, statistics, and UI elements using alpha-enabled
    surfaces and vectorized operations where possible.

    Attributes:
        surface: Main Pygame surface where all visualization is composited
        config: Simulation configuration parameters containing rendering options
        particle_surface: Alpha-enabled surface for optimized particle rendering
        font: Pygame font for text rendering of statistics and metrics
    """

    def __init__(self, surface: pygame.Surface, config: SimulationConfig) -> None:
        """
        Initialize the renderer with drawing surfaces and font resources.

        Args:
            surface: The target Pygame surface for all visualization output
            config: Configuration parameters controlling rendering properties
        """
        self.surface: pygame.Surface = surface
        self.config: SimulationConfig = config

        # Create alpha-enabled surface for composite particle rendering
        self.particle_surface: pygame.Surface = pygame.Surface(
            self.surface.get_size(), flags=pygame.SRCALPHA
        ).convert_alpha()

        # Initialize font renderer for statistics display
        pygame.font.init()
        self.font: pygame.font.Font = pygame.font.SysFont("Arial", 20)

    def draw_component(
        self,
        x: float,
        y: float,
        color: Tuple[int, int, int],
        energy: float,
        speed_factor: float,
    ) -> None:
        """
        Draw a single cellular component with energy-based visual properties.

        Renders a particle as a circle with color intensity modulated by its
        energy level and speed characteristics.

        Args:
            x: Horizontal position of the component
            y: Vertical position of the component
            color: Base RGB color tuple for the component (r,g,b)
            energy: Current energy level affecting brightness
            speed_factor: Speed trait affecting color intensity
        """
        # Normalize energy to [0,1] range for intensity calculations
        health: float = min(100.0, max(0.0, energy))
        intensity_factor: float = health / 100.0

        # Calculate color with energy-modulated brightness
        # Higher energy/speed creates brighter particles, lower creates dimmer ones
        c: Tuple[int, int, int] = (
            min(
                255,
                int(
                    color[0] * intensity_factor * speed_factor
                    + (1 - intensity_factor) * 100
                ),
            ),
            min(
                255,
                int(
                    color[1] * intensity_factor * speed_factor
                    + (1 - intensity_factor) * 100
                ),
            ),
            min(
                255,
                int(
                    color[2] * intensity_factor * speed_factor
                    + (1 - intensity_factor) * 100
                ),
            ),
        )

        # Render the cellular component as a circle
        pygame.draw.circle(
            self.particle_surface, c, (int(x), int(y)), int(self.config.particle_size)
        )

    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """
        Draw all alive components of a specific cellular type.

        Uses vectorized operations to efficiently identify and render
        only the active components.

        Args:
            ct: Cellular type data containing component properties
        """
        # Get indices of all living components for efficient iteration
        alive_indices: np.ndarray = np.where(ct.alive)[0]

        # Draw each living component with its properties
        for idx in alive_indices:
            self.draw_component(
                ct.x[idx], ct.y[idx], ct.color, ct.energy[idx], ct.speed_factor[idx]
            )

    def render(self, stats: Dict[str, float]) -> None:
        """
        Render all visualization elements to the main surface.

        Composites the particle layer with the main surface and renders
        statistical information as an overlay.

        Args:
            stats: Dictionary of simulation statistics to display
                  (fps, total_species, total_particles)
        """
        # Composite the particle surface onto the main surface
        self.surface.blit(self.particle_surface, (0, 0))

        # Clear the particle surface for the next frame
        self.particle_surface.fill((0, 0, 0, 0))

        # Format and render statistics text
        stats_text: str = (
            f"FPS: {stats.get('fps', 0):.2f} | "
            f"Species: {stats.get('total_species', 0)} | "
            f"Particles: {stats.get('total_particles', 0)}"
        )

        # Create and position the statistics text surface
        text_surface: pygame.Surface = self.font.render(
            stats_text, True, (255, 255, 255)
        )
        self.surface.blit(text_surface, (10, 10))
