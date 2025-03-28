from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pygame
from gp_config import SimulationConfig
from gp_manager import CellularTypeManager
from gp_renderer import Renderer
from gp_rules import InteractionRules
from gp_types import CellularTypeData
from gp_utility import apply_synergy, generate_vibrant_colors, give_take_interaction
from scipy.spatial import cKDTree

###############################################################
# Cellular Automata (Main Simulation)
###############################################################


class CellularAutomata:
    """
    The main simulation controller implementing cellular automata dynamics.

    Orchestrates all simulation aspects including initialization, frame updates,
    inter-type interactions, clustering behaviors, reproduction, and visualization.
    Manages the full lifecycle of particles with optimized vectorized operations.

    Attributes:
        config: Master configuration parameters
        screen: Pygame display surface
        clock: Timing controller for frame rate management
        frame_count: Current simulation frame number
        run_flag: Boolean controlling main loop execution
        edge_buffer: Distance from screen edges for boundary calculations
        colors: RGB color tuples for each cellular type
        type_manager: Controller for all cellular type data
        rules_manager: Manager of interaction rules between types
        renderer: Handles visual representation of particles
        species_count: Dictionary tracking population by species ID
        screen_bounds: NumPy array of screen boundary coordinates
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize the simulation environment with specified configuration.

        Sets up display, managers, and initial particle population. Creates
        cellular types with randomized properties based on configuration,
        with mass-based physics applied to a subset of types.

        Args:
            config: Configuration parameters controlling all simulation aspects
        """
        # Core system initialization
        self.config: SimulationConfig = config
        pygame.init()

        # Display setup
        display_info: pygame.display.Info = pygame.display.Info()
        screen_width: int = display_info.current_w
        screen_height: int = display_info.current_h
        self.screen: pygame.Surface = pygame.display.set_mode(
            (screen_width, screen_height),
            pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption("Emergent Cellular Automata Simulation")

        # Simulation control variables
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.frame_count: int = 0
        self.run_flag: bool = True

        # Calculate screen boundaries with buffer
        self.edge_buffer: float = 0.05 * max(screen_width, screen_height)
        self.screen_bounds: np.ndarray = np.array(
            [
                self.edge_buffer,  # Left bound
                screen_width - self.edge_buffer,  # Right bound
                self.edge_buffer,  # Top bound
                screen_height - self.edge_buffer,  # Bottom bound
            ]
        )

        # Generate particle type colors and physics properties
        self.colors: List[Tuple[int, int, int]] = generate_vibrant_colors(
            self.config.n_cell_types
        )
        n_mass_types: int = int(
            self.config.mass_based_fraction * self.config.n_cell_types
        )
        mass_based_type_indices: List[int] = list(range(n_mass_types))

        # Initialize managers for particle types and interactions
        self.type_manager: CellularTypeManager = CellularTypeManager(
            self.config, self.colors, mass_based_type_indices
        )

        # Generate mass values for mass-based particle types
        mass_values: np.ndarray = np.random.uniform(
            self.config.mass_range[0], self.config.mass_range[1], n_mass_types
        )

        # Create and initialize all cellular types
        for i in range(self.config.n_cell_types):
            ct = CellularTypeData(
                type_id=i,
                color=self.colors[i],
                n_particles=self.config.particles_per_type,
                window_width=screen_width,
                window_height=screen_height,
                initial_energy=self.config.initial_energy,
                max_age=self.config.max_age,
                mass=mass_values[i] if i < n_mass_types else None,
                base_velocity_scale=self.config.base_velocity_scale,
            )
            self.type_manager.add_cellular_type_data(ct)

        # Initialize interaction rules and rendering system
        self.rules_manager: InteractionRules = InteractionRules(
            self.config, mass_based_type_indices
        )
        self.renderer: Renderer = Renderer(self.screen, self.config)

        # Initialize species tracking
        self.species_count: Dict[int, int] = defaultdict(int)
        self.update_species_count()

    def update_species_count(self) -> None:
        """
        Update the count of unique species across all cellular types.

        Clears existing counts and recalculates population sizes for each
        species ID found across all cellular types.
        """
        self.species_count.clear()
        for ct in self.type_manager.cellular_types:
            unique, counts = np.unique(ct.species_id, return_counts=True)
            for species, count in zip(unique, counts):
                self.species_count[species] += count

    def main_loop(self) -> None:
        """
        Execute the main simulation loop until termination conditions are met.

        Each frame performs:
        1. Event handling (exit conditions, user input)
        2. Interaction rule evolution
        3. Inter-type interactions (forces, energy transfers)
        4. Clustering behaviors within types
        5. Reproduction and death management
        6. Boundary handling and rendering
        7. Performance monitoring and population control

        Loop exits on ESC key, window close, or reaching max_frames.
        """
        while self.run_flag:
            # Update frame counter and check termination conditions
            self.frame_count += 1
            if self.config.max_frames > 0 and self.frame_count > self.config.max_frames:
                self.run_flag = False

            # Handle Pygame events (window close, key presses)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    self.run_flag = False
                    break

            # Evolve interaction parameters periodically
            self.rules_manager.evolve_parameters(self.frame_count)

            # Clear the display with background color
            self.screen.fill((69, 69, 69))

            # Apply all inter-type interactions and updates
            self.apply_all_interactions()

            # Apply clustering within each cellular type
            for ct in self.type_manager.cellular_types:
                self.apply_clustering(ct)

            # Handle reproduction and death across all types
            self.type_manager.reproduce()
            self.type_manager.remove_dead_in_all_types()
            self.update_species_count()

            # Render all cellular types to the display
            for ct in self.type_manager.cellular_types:
                self.renderer.draw_cellular_type(ct)

            # Handle boundary reflections for all types
            self.handle_boundary_reflections()

            # Prepare statistics for display
            stats: Dict[str, float] = {
                "fps": self.clock.get_fps(),
                "total_species": len(self.species_count),
                "total_particles": sum(self.species_count.values()),
            }

            # Render UI elements with current statistics
            self.renderer.render(stats)
            pygame.display.flip()

            # Limit frame rate and get current FPS
            current_fps: float = self.clock.tick(120)
            self.display_fps(self.screen, current_fps)

            # Adaptive population control based on performance
            if self.frame_count % 10 == 0:
                if (
                    any(ct.x.size > 50 for ct in self.type_manager.cellular_types)
                    or current_fps <= 60
                ):
                    self.cull_oldest_particles()

        # Clean up Pygame resources on exit
        pygame.quit()

    def display_fps(self, surface: pygame.Surface, fps: float) -> None:
        """
        Display the current FPS counter in the top-left corner of the screen.

        Args:
            surface: Pygame surface to render the FPS text on
            fps: Current frames per second value
        """
        font: pygame.font.Font = pygame.font.Font(None, 36)
        fps_text: pygame.Surface = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
        surface.blit(fps_text, (10, 10))

    def apply_all_interactions(self) -> None:
        """
        Process all type-to-type interactions defined in the rules matrix.

        Iterates through all interaction rule parameters and applies them
        between the corresponding cellular type pairs.
        """
        for i, j, params in self.rules_manager.rules:
            self.apply_interaction_between_types(i, j, params)

    def apply_interaction_between_types(
        self, i: int, j: int, params: Dict[str, Union[float, bool, np.ndarray]]
    ) -> None:
        """
        Apply physics, energy transfers, and synergy between two cellular types.

        Calculates forces between particles of different types based on distance,
        applies potential and gravitational forces, handles predator-prey energy
        transfers, and processes cooperative energy sharing (synergy).

        Args:
            i: Index of the first cellular type
            j: Index of the second cellular type
            params: Parameters controlling interaction physics and forces
        """
        # Retrieve cellular types and their interaction properties
        ct_i: CellularTypeData = self.type_manager.get_cellular_type_by_id(i)
        ct_j: CellularTypeData = self.type_manager.get_cellular_type_by_id(j)
        synergy_factor: float = self.rules_manager.synergy_matrix[i, j]
        is_giver: bool = self.rules_manager.give_take_matrix[i, j]

        # Skip if either type has no particles
        n_i: int = ct_i.x.size
        n_j: int = ct_j.x.size
        if n_i == 0 or n_j == 0:
            return

        # Configure gravity parameters for mass-based types
        if params.get("use_gravity", False):
            if (
                ct_i.mass_based
                and ct_i.mass is not None
                and ct_j.mass_based
                and ct_j.mass is not None
            ):
                params["m_a"] = ct_i.mass
                params["m_b"] = ct_j.mass
            else:
                params["use_gravity"] = False

        # Calculate pairwise distances using vectorized operations
        dx: np.ndarray = ct_i.x[:, np.newaxis] - ct_j.x
        dy: np.ndarray = ct_i.y[:, np.newaxis] - ct_j.y
        dist_sq: np.ndarray = dx * dx + dy * dy

        # Create interaction mask for particles within range
        within_range: np.ndarray = (dist_sq > 0.0) & (
            dist_sq <= params["max_dist"] ** 2
        )

        # Get indices of interacting particle pairs
        indices: Tuple[np.ndarray, np.ndarray] = np.where(within_range)
        if len(indices[0]) == 0:
            return

        # Calculate distances for interacting particles
        dist: np.ndarray = np.sqrt(dist_sq[indices])

        # Initialize force components
        fx: np.ndarray = np.zeros_like(dist)
        fy: np.ndarray = np.zeros_like(dist)

        # Calculate potential-based forces
        if params.get("use_potential", True):
            pot_strength: float = params.get("potential_strength", 1.0)
            F_pot: np.ndarray = pot_strength / dist
            fx += F_pot * (dx[indices] / dist)
            fy += F_pot * (dy[indices] / dist)

        # Calculate gravitational forces if applicable
        if params.get("use_gravity", False):
            gravity_factor: float = params.get("gravity_factor", 1.0)
            F_grav: np.ndarray = (
                gravity_factor
                * (params["m_a"][indices[0]] * params["m_b"][indices[1]])
                / dist_sq[indices]
            )
            fx += F_grav * (dx[indices] / dist)
            fy += F_grav * (dy[indices] / dist)

        # Apply calculated forces to velocities
        np.add.at(ct_i.vx, indices[0], fx)
        np.add.at(ct_i.vy, indices[0], fy)

        # Handle predator-prey energy transfers (give-take)
        if is_giver:
            # Find pairs within predation range
            give_take_within: np.ndarray = (
                dist_sq[indices] <= self.config.predation_range**2
            )
            give_take_indices: Tuple[np.ndarray, np.ndarray] = (
                indices[0][give_take_within],
                indices[1][give_take_within],
            )

            if give_take_indices[0].size > 0:
                # Extract energy and mass values for transfer
                giver_energy: np.ndarray = ct_i.energy[give_take_indices[0]]
                receiver_energy: np.ndarray = ct_j.energy[give_take_indices[1]]
                giver_mass: Optional[np.ndarray] = (
                    ct_i.mass[give_take_indices[0]]
                    if ct_i.mass_based and ct_i.mass is not None
                    else None
                )
                receiver_mass: Optional[np.ndarray] = (
                    ct_j.mass[give_take_indices[1]]
                    if ct_j.mass_based and ct_j.mass is not None
                    else None
                )

                # Process energy and mass transfers
                updated: Tuple[
                    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]
                ] = give_take_interaction(
                    giver_energy,
                    receiver_energy,
                    giver_mass,
                    receiver_mass,
                    self.config,
                )

                # Update energy and mass values after transfer
                ct_i.energy[give_take_indices[0]] = updated[0]
                ct_j.energy[give_take_indices[1]] = updated[1]

                if ct_i.mass_based and ct_i.mass is not None and updated[2] is not None:
                    ct_i.mass[give_take_indices[0]] = updated[2]
                if ct_j.mass_based and ct_j.mass is not None and updated[3] is not None:
                    ct_j.mass[give_take_indices[1]] = updated[3]

        # Handle synergy (cooperative energy sharing)
        if synergy_factor > 0.0 and self.config.synergy_range > 0.0:
            # Find pairs within synergy range
            synergy_within: np.ndarray = (
                dist_sq[indices] <= self.config.synergy_range**2
            )
            synergy_indices: Tuple[np.ndarray, np.ndarray] = (
                indices[0][synergy_within],
                indices[1][synergy_within],
            )

            if synergy_indices[0].size > 0:
                # Extract energy values for redistribution
                energyA: np.ndarray = ct_i.energy[synergy_indices[0]]
                energyB: np.ndarray = ct_j.energy[synergy_indices[1]]

                # Apply energy sharing based on synergy factor
                new_energyA, new_energyB = apply_synergy(
                    energyA, energyB, synergy_factor
                )
                ct_i.energy[synergy_indices[0]] = new_energyA
                ct_j.energy[synergy_indices[1]] = new_energyB

        # Apply friction to velocities
        friction_factor: float = 1.0 - self.config.friction
        ct_i.vx *= friction_factor
        ct_i.vy *= friction_factor

        # Apply thermal noise (random motion)
        thermal_noise: np.ndarray = (
            np.random.uniform(-0.5, 0.5, n_i) * self.config.global_temperature
        )
        ct_i.vx += thermal_noise
        ct_i.vy += thermal_noise

        # Update positions based on velocities
        ct_i.x += ct_i.vx
        ct_i.y += ct_i.vy

        # Handle boundary conditions and state updates
        self.handle_boundary_reflections(ct_i)
        ct_i.age_components()
        ct_i.update_states()
        ct_i.update_alive()

    def handle_boundary_reflections(
        self, ct: Optional[CellularTypeData] = None
    ) -> None:
        """
        Reflect particles at screen boundaries and clamp positions to valid range.

        When particles reach screen edges, their velocities are reversed in the
        appropriate dimension and positions are constrained to remain within bounds.

        Args:
            ct: Specific cellular type to process; if None, processes all types
        """
        # Determine which cellular types to process
        cellular_types: List[CellularTypeData] = (
            [ct] if ct is not None else self.type_manager.cellular_types
        )

        for ct in cellular_types:
            if ct.x.size == 0:
                continue

            # Create boolean masks for boundary violations in each direction
            left_mask: np.ndarray = ct.x < self.screen_bounds[0]
            right_mask: np.ndarray = ct.x > self.screen_bounds[1]
            top_mask: np.ndarray = ct.y < self.screen_bounds[2]
            bottom_mask: np.ndarray = ct.y > self.screen_bounds[3]

            # Reflect velocities for particles at boundaries
            ct.vx[left_mask | right_mask] *= -1
            ct.vy[top_mask | bottom_mask] *= -1

            # Clamp positions to remain within screen bounds
            np.clip(ct.x, self.screen_bounds[0], self.screen_bounds[1], out=ct.x)
            np.clip(ct.y, self.screen_bounds[2], self.screen_bounds[3], out=ct.y)

    def cull_oldest_particles(self) -> None:
        """
        Remove oldest particles from populated cellular types to maintain performance.

        When a cellular type exceeds a size threshold (500 particles), removes its
        oldest particle to prevent performance degradation. This helps maintain
        framerate while preserving the overall ecological balance.
        """
        for ct in self.type_manager.cellular_types:
            # Skip types with reasonable population sizes
            if ct.x.size < 500:
                continue

            # Identify the oldest particle
            oldest_idx: int = np.argmax(ct.age)

            # Create a mask excluding the oldest particle
            keep_mask: np.ndarray = np.ones(ct.x.size, dtype=bool)
            keep_mask[oldest_idx] = False

            # Apply the mask to all component arrays
            ct.x = ct.x[keep_mask]
            ct.y = ct.y[keep_mask]
            ct.vx = ct.vx[keep_mask]
            ct.vy = ct.vy[keep_mask]
            ct.energy = ct.energy[keep_mask]
            ct.alive = ct.alive[keep_mask]
            ct.age = ct.age[keep_mask]
            ct.energy_efficiency = ct.energy_efficiency[keep_mask]
            ct.speed_factor = ct.speed_factor[keep_mask]
            ct.interaction_strength = ct.interaction_strength[keep_mask]
            ct.perception_range = ct.perception_range[keep_mask]
            ct.reproduction_rate = ct.reproduction_rate[keep_mask]
            ct.synergy_affinity = ct.synergy_affinity[keep_mask]
            ct.colony_factor = ct.colony_factor[keep_mask]
            ct.drift_sensitivity = ct.drift_sensitivity[keep_mask]
            ct.species_id = ct.species_id[keep_mask]
            ct.parent_id = ct.parent_id[keep_mask]

            # Handle mass array for mass-based types
            if ct.mass_based and ct.mass is not None:
                ct.mass = ct.mass[keep_mask]

    def add_global_energy(self) -> None:
        """
        Increase energy levels of all particles across the simulation.

        Adds 10% to current energy levels of all particles across all types,
        clamping to maximum allowed value of 200 units. Simulates environmental
        energy input into the system.
        """
        for ct in self.type_manager.cellular_types:
            ct.energy = np.clip(ct.energy * 1.1, 0.0, 200.0)

    def apply_clustering(self, ct: CellularTypeData) -> None:
        """
        Apply flocking behavior within a cellular type using the Boids algorithm.

        Implements three steering behaviors:
        1. Alignment: Match velocity with nearby neighbors
        2. Cohesion: Move toward the center of nearby neighbors
        3. Separation: Avoid crowding nearby neighbors

        Uses KD-Tree for efficient nearest neighbor queries.

        Args:
            ct: Cellular type to apply clustering behavior to
        """
        # Skip if too few particles for meaningful clustering
        n: int = ct.x.size
        if n < 2:
            return

        # Build KD-Tree for efficient neighbor searching
        positions: np.ndarray = np.column_stack((ct.x, ct.y))
        tree: cKDTree = cKDTree(positions)

        # Query all neighbors within cluster radius
        indices: List[List[int]] = tree.query_ball_tree(
            tree, self.config.cluster_radius
        )

        # Pre-allocate velocity change arrays
        dvx: np.ndarray = np.zeros(n)
        dvy: np.ndarray = np.zeros(n)

        # Process each particle and its neighbors
        for idx, neighbor_indices in enumerate(indices):
            # Filter out self and dead neighbors
            neighbor_indices = [i for i in neighbor_indices if i != idx and ct.alive[i]]
            if not neighbor_indices:
                continue

            # Extract neighbor positions and velocities
            neighbor_positions: np.ndarray = positions[neighbor_indices]
            neighbor_velocities: np.ndarray = np.column_stack(
                (ct.vx[neighbor_indices], ct.vy[neighbor_indices])
            )

            # 1. Alignment - Match velocity with nearby neighbors
            avg_velocity: np.ndarray = np.mean(neighbor_velocities, axis=0)
            alignment: np.ndarray = (
                avg_velocity - np.array([ct.vx[idx], ct.vy[idx]])
            ) * self.config.alignment_strength

            # 2. Cohesion - Move toward the center of nearby neighbors
            center: np.ndarray = np.mean(neighbor_positions, axis=0)
            cohesion: np.ndarray = (
                center - positions[idx]
            ) * self.config.cohesion_strength

            # 3. Separation - Avoid crowding nearby neighbors
            separation: np.ndarray = (
                positions[idx] - np.mean(neighbor_positions, axis=0)
            ) * self.config.separation_strength

            # Combine all forces and store for later application
            total_force: np.ndarray = alignment + cohesion + separation
            dvx[idx] = total_force[0]
            dvy[idx] = total_force[1]

        # Apply accumulated velocity changes to all particles at once
        ct.vx += dvx
        ct.vy += dvy
