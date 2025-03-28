"""Gene Particles Rendering System.

Provides optimized visualization tools for cellular automata simulation with
energy-modulated aesthetics, real-time statistical displays, and alpha-composited
rendering for maximum performance and visual clarity.
"""

from typing import Dict

import numpy as np
import pygame

from game_forge.src.gene_particles.gp_config import SimulationConfig
from game_forge.src.gene_particles.gp_types import CellularTypeData
from game_forge.src.gene_particles.gp_utility import (
    BoolArray,
    ColorRGB,
    FloatArray,
    IntArray,
)

###############################################################
# Visual Enhancement Constants
###############################################################

# Text colors for rendering different statistics with visual distinction
FPS_COLOR: ColorRGB = (50, 255, 50)  # Bright green for FPS
SPECIES_COLOR: ColorRGB = (255, 180, 0)  # Orange for species count
PARTICLES_COLOR: ColorRGB = (100, 200, 255)  # Light blue for particle count

# Background opacity for stats display (0-255)
STATS_BG_ALPHA: int = 120

###############################################################
# Renderer Class
###############################################################


class Renderer:
    """High-performance visualization engine for cellular components.

    Handles rendering of cellular components on a Pygame surface with optimized visualization
    techniques including alpha compositing, energy-modulated coloration, and vectorized
    drawing operations for maximum performance.

    Attributes:
        surface: Main Pygame surface where all visualization is composited
        config: Simulation configuration parameters containing rendering options
        particle_surface: Alpha-enabled surface for optimized particle rendering
        stats_surface: Semi-transparent surface for statistics display
        font: Pygame font for text rendering of statistics and metrics
    """

    def __init__(self, surface: pygame.Surface, config: SimulationConfig) -> None:
        """Initialize the rendering engine with required surfaces and resources.

        Creates alpha-enabled composition surfaces for layered rendering and
        initializes font resources for aesthetically pleasing statistics display.

        Args:
            surface: The target Pygame surface for all visualization output
            config: Configuration parameters controlling rendering properties
        """
        self.surface: pygame.Surface = surface
        self.config: SimulationConfig = config

        # Window dimensions for convenience
        self.width, self.height = self.surface.get_size()

        # Create alpha-enabled surface for composite particle rendering
        self.particle_surface: pygame.Surface = pygame.Surface(
            (self.width, self.height), flags=pygame.SRCALPHA
        ).convert_alpha()

        # Create semi-transparent surface for statistics display
        self.stats_surface: pygame.Surface = pygame.Surface(
            (self.width, 60), flags=pygame.SRCALPHA
        ).convert_alpha()

        # Initialize font renderer for statistics display
        pygame.font.init()
        self.font: pygame.font.Font = pygame.font.SysFont("Arial", 20)

    def draw_component(
        self,
        x: float,
        y: float,
        color: ColorRGB,
        energy: float,
        speed_factor: float,
    ) -> None:
        """Draw a single cellular component with energy-based visual properties.

        Renders a particle as a circle with color intensity modulated by its
        energy level and speed characteristics. Higher energy particles appear
        brighter and more vibrant.

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
        r: int = min(
            255,
            int(
                color[0] * intensity_factor * speed_factor
                + (1 - intensity_factor) * 100
            ),
        )
        g: int = min(
            255,
            int(
                color[1] * intensity_factor * speed_factor
                + (1 - intensity_factor) * 100
            ),
        )
        b: int = min(
            255,
            int(
                color[2] * intensity_factor * speed_factor
                + (1 - intensity_factor) * 100
            ),
        )

        # Energy also slightly affects particle size for visual differentiation
        size_factor: float = 0.8 + (intensity_factor * 0.4)
        particle_size: int = int(self.config.particle_size * size_factor)

        # Pack the calculated RGB values into a color tuple
        c: ColorRGB = (r, g, b)

        # Render the cellular component as a circle
        pygame.draw.circle(self.particle_surface, c, (int(x), int(y)), particle_size)

    def draw_cellular_type(self, ct: CellularTypeData) -> None:
        """Draw all alive components of a specific cellular type.

        Uses vectorized operations to efficiently identify and render
        only the active components. Extracts array indices of living cells
        to avoid wasting computation on inactive components.

        Args:
            ct: Cellular type data containing component properties and state
        """
        # Get indices of all living components for efficient iteration
        # Explicitly type the arrays to avoid type inference issues
        alive_array: BoolArray = ct.alive  # Already properly typed via Protocol
        alive_indices: IntArray = np.where(alive_array)[0].astype(np.int_)

        # Get typed references to the component arrays
        x_array: FloatArray = ct.x
        y_array: FloatArray = ct.y
        energy_array: FloatArray = ct.energy
        speed_factor_array: FloatArray = ct.speed_factor

        # Draw each living component with its properties
        for idx in alive_indices:
            # Safely extract scalar values from arrays at the specified index
            x_pos: float = float(x_array[idx])
            y_pos: float = float(y_array[idx])
            energy_val: float = float(energy_array[idx])
            speed_factor_val: float = float(speed_factor_array[idx])

            # Draw individual component with extracted scalar values
            self.draw_component(x_pos, y_pos, ct.color, energy_val, speed_factor_val)

    def _render_statistics(self, stats: Dict[str, float]) -> None:
        """Render simulation statistics with visually distinct styling.

        Creates a semi-transparent overlay with color-coded statistics
        for improved readability and visual appeal.

        Args:
            stats: Dictionary of simulation statistics to display
                  (fps, total_species, total_particles)
        """
        # Clear the stats surface with a semi-transparent background
        self.stats_surface.fill((20, 20, 30, STATS_BG_ALPHA))

        # Extract statistics with defaults for safety
        fps: float = stats.get("fps", 0.0)
        species_count: int = int(stats.get("total_species", 0))
        particle_count: int = int(stats.get("total_particles", 0))

        # Render each statistic with its own color and position
        fps_text: pygame.Surface = self.font.render(f"FPS: {fps:.1f}", True, FPS_COLOR)
        species_text: pygame.Surface = self.font.render(
            f"Species: {species_count}", True, SPECIES_COLOR
        )
        particles_text: pygame.Surface = self.font.render(
            f"Particles: {particle_count}", True, PARTICLES_COLOR
        )

        # Position and blit each text element
        self.stats_surface.blit(fps_text, (20, 15))
        self.stats_surface.blit(species_text, (150, 15))
        self.stats_surface.blit(particles_text, (300, 15))

        # Apply the stats surface to the main surface
        self.surface.blit(self.stats_surface, (0, 0))

    def render(self, stats: Dict[str, float]) -> None:
        """Render all visualization elements to the main surface.

        Composites the particle layer with the main surface and renders
        statistical information as an overlay. Clears the particle buffer
        after each frame to prepare for the next rendering cycle.

        Args:
            stats: Dictionary of simulation statistics to display
                  (fps, total_species, total_particles)
        """
        # Composite the particle surface onto the main surface
        self.surface.blit(self.particle_surface, (0, 0))

        # Clear the particle surface for the next frame
        self.particle_surface.fill((0, 0, 0, 0))

        # Render statistics with enhanced visual styling
        self._render_statistics(stats)
