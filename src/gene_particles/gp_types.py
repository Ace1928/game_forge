from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, TypeVar

import numpy as np
from gp_config import SimulationConfig
from gp_utility import random_xy
from numpy.typing import NDArray
from scipy.spatial import cKDTree

# Type aliases for enhanced readability
Vector2D = Tuple[float, float]  # (x, y) coordinate pair
ColorRGB = Tuple[int, int, int]  # (r, g, b) color values 0-255
# Type aliases for improved readability
FloatArray = NDArray[np.float64]
T = TypeVar("T")
Range = Tuple[float, float]


class TraitType(Enum):
    """Genetic trait categories for organizational and validation purposes."""

    MOVEMENT = auto()  # Traits affecting particle motion
    INTERACTION = auto()  # Traits affecting inter-particle forces
    PERCEPTION = auto()  # Traits affecting sensing and awareness
    REPRODUCTION = auto()  # Traits affecting breeding behaviors
    SOCIAL = auto()  # Traits affecting group dynamics
    ADAPTATION = auto()  # Traits affecting evolutionary dynamics


@dataclass(frozen=True)
class TraitDefinition:
    """
    Immutable definition of a genetic trait's properties and constraints.

    Attributes:
         name: Unique identifier for the trait
         type: Categorical classification of trait purpose
         range: Valid minimum/maximum values
         description: Human-readable explanation of trait function
         default: Starting value for initialization
    """

    name: str
    type: TraitType
    range: Range
    description: str
    default: float


class CellularTypeData:
    """
    Represents a cellular type with multiple cellular components.
    Manages positions, velocities, energy, mass, and genetic traits of components.
    """

    def __init__(
        self,
        type_id: int,
        color: Tuple[int, int, int],
        n_particles: int,
        window_width: int,
        window_height: int,
        initial_energy: float,
        max_age: float = np.inf,
        mass: Optional[float] = None,
        base_velocity_scale: float = 1.0,
        energy_efficiency: Optional[float] = None,
        gene_traits: List[str] = [
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
        ],
        gene_mutation_rate: float = 0.05,
        gene_mutation_range: Tuple[float, float] = (-0.1, 0.1),
        min_energy: float = 0.0,
        max_energy: float = 1000.0,
        min_mass: float = 0.1,
        max_mass: float = 10.0,
        min_velocity: float = -10.0,
        max_velocity: float = 10.0,
        min_perception: float = 10.0,
        max_perception: float = 300.0,
        min_reproduction: float = 0.05,
        max_reproduction: float = 1.0,
        min_synergy: float = 0.0,
        max_synergy: float = 2.0,
        min_colony: float = 0.0,
        max_colony: float = 1.0,
        min_drift: float = 0.0,
        max_drift: float = 2.0,
        min_energy_efficiency: float = -0.3,
        max_energy_efficiency: float = 2.5,
    ):
        """
        Initialize a CellularTypeData instance with given parameters.

        Parameters:
        -----------
        type_id : int
            Unique identifier for the cellular type.
        color : Tuple[int, int, int]
            RGB color tuple for rendering cellular components of this type.
        n_particles : int
            Initial number of cellular components in this type.
        window_width : int
            Width of the simulation window in pixels.
        window_height : int
            Height of the simulation window in pixels.
        initial_energy : float
            Initial energy assigned to each cellular component.
        max_age : float, default=np.inf
            Maximum age a cellular component can reach before dying.
        mass : Optional[float], default=None
            Mass of cellular components if the type is mass-based.
        base_velocity_scale : float, default=1.0
            Base scaling factor for initial velocities of cellular components.
        energy_efficiency : Optional[float], default=None
            Initial energy efficiency trait of cellular components. If None, randomly initialized within range.
        gene_traits : List[str], default=["speed_factor", "interaction_strength", "perception_range", "reproduction_rate", "synergy_affinity", "colony_factor", "drift_sensitivity"]
            List of gene trait names.
        gene_mutation_rate : float, default=0.05
            Base mutation rate for gene traits (0.0 to 1.0).
        gene_mutation_range : Tuple[float, float], default=(-0.1, 0.1)
            Range for gene trait mutations (units: arbitrary).
        min_energy : float, default=0.0
            Minimum allowed energy value.
        max_energy : float, default=1000.0
            Maximum allowed energy value.
        min_mass : float, default=0.1
            Minimum allowed mass value.
        max_mass : float, default=10.0
            Maximum allowed mass value.
        min_velocity : float, default=-10.0
            Minimum allowed velocity value.
        max_velocity : float, default=10.0
            Maximum allowed velocity value.
        min_perception : float, default=10.0
            Minimum allowed perception range.
        max_perception : float, default=300.0
            Maximum allowed perception range.
        min_reproduction : float, default=0.05
            Minimum allowed reproduction rate.
        max_reproduction : float, default=1.0
            Maximum allowed reproduction rate.
        min_synergy : float, default=0.0
            Minimum allowed synergy affinity.
        max_synergy : float, default=2.0
            Maximum allowed synergy affinity.
        min_colony : float, default=0.0
            Minimum allowed colony factor.
        max_colony : float, default=1.0
            Maximum allowed colony factor.
        min_drift : float, default=0.0
            Maximum allowed drift sensitivity.
        """
        # Store metadata
        self.type_id: int = type_id  # Unique ID for the cellular type
        self.color: Tuple[int, int, int] = color  # RGB color for rendering
        self.mass_based: bool = (
            mass is not None
        )  # Flag indicating if type is mass-based

        # Store parameter bounds
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_perception = min_perception
        self.max_perception = max_perception
        self.min_reproduction = min_reproduction
        self.max_reproduction = max_reproduction
        self.min_synergy = min_synergy
        self.max_synergy = max_synergy
        self.min_colony = min_colony
        self.max_colony = max_colony
        self.min_drift = min_drift
        self.max_drift = max_drift
        self.min_energy_efficiency = min_energy_efficiency
        self.max_energy_efficiency = max_energy_efficiency

        # Initialize cellular component positions randomly within the window
        coords = random_xy(
            window_width, window_height, n_particles
        )  # Generate random (x,y) positions
        self.x: np.ndarray = coords[:, 0]  # X positions as float array
        self.y: np.ndarray = coords[:, 1]  # Y positions as float array

        # Initialize energy efficiency trait
        if energy_efficiency is None:
            # Randomly initialize energy efficiency within the defined range
            self.energy_efficiency: np.ndarray = np.random.uniform(
                self.min_energy_efficiency, self.max_energy_efficiency, n_particles
            )
        else:
            # Set a fixed energy efficiency if provided
            self.energy_efficiency: np.ndarray = np.full(
                n_particles, energy_efficiency, dtype=float
            )

        # Calculate velocity scaling based on energy efficiency and speed factor
        velocity_scaling = (
            base_velocity_scale / self.energy_efficiency
        )  # Higher efficiency -> lower speed

        # Initialize cellular component velocities with random values scaled by velocity_scaling
        self.vx: np.ndarray = np.clip(
            np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
            self.min_velocity,
            self.max_velocity,
        ).astype(
            float
        )  # X velocities
        self.vy: np.ndarray = np.clip(
            np.random.uniform(-0.5, 0.5, n_particles) * velocity_scaling,
            self.min_velocity,
            self.max_velocity,
        ).astype(
            float
        )  # Y velocities

        # Initialize energy levels for all cellular components
        self.energy: np.ndarray = np.clip(
            np.full(n_particles, initial_energy, dtype=float),
            self.min_energy,
            self.max_energy,
        )  # Energy levels

        # Initialize mass if type is mass-based
        if self.mass_based:
            if mass is None or mass <= 0.0:
                # Mass must be positive for mass-based types
                raise ValueError("Mass must be positive for mass-based cellular types.")
            self.mass: np.ndarray = np.clip(
                np.full(n_particles, mass, dtype=float), self.min_mass, self.max_mass
            )  # Mass values
        else:
            self.mass = None  # Mass is None for massless types

        # Initialize alive status and age for cellular components
        self.alive: np.ndarray = np.ones(
            n_particles, dtype=bool
        )  # All components start as alive
        self.age: np.ndarray = np.zeros(n_particles, dtype=float)  # Initial age is 0
        self.max_age: float = max_age  # Maximum age before death

        # Initialize gene traits
        # Genes influence behaviors and can mutate during reproduction
        self.speed_factor: np.ndarray = np.random.uniform(
            0.5, 1.5, n_particles
        )  # Speed scaling factors
        self.interaction_strength: np.ndarray = np.random.uniform(
            0.5, 1.5, n_particles
        )  # Interaction force scaling factors
        self.perception_range: np.ndarray = np.clip(
            np.random.uniform(50.0, 150.0, n_particles),
            self.min_perception,
            self.max_perception,
        )  # Perception ranges for interactions
        self.reproduction_rate: np.ndarray = np.clip(
            np.random.uniform(0.1, 0.5, n_particles),
            self.min_reproduction,
            self.max_reproduction,
        )  # Reproduction rates
        self.synergy_affinity: np.ndarray = np.clip(
            np.random.uniform(0.5, 1.5, n_particles), self.min_synergy, self.max_synergy
        )  # Synergy affinity factors
        self.colony_factor: np.ndarray = np.clip(
            np.random.uniform(0.0, 1.0, n_particles), self.min_colony, self.max_colony
        )  # Colony formation factors
        self.drift_sensitivity: np.ndarray = np.clip(
            np.random.uniform(0.5, 1.5, n_particles), self.min_drift, self.max_drift
        )  # Evolutionary drift sensitivity

        # Gene mutation parameters
        self.gene_mutation_rate: float = gene_mutation_rate  # Mutation rate for genes
        self.gene_mutation_range: Tuple[float, float] = (
            gene_mutation_range  # Mutation range for genes
        )

        # Speciation parameters
        self.species_id: np.ndarray = np.full(
            n_particles, type_id, dtype=int
        )  # Species IDs, initially same as type_id

        # Lineage tracking
        self.parent_id: np.ndarray = np.full(
            n_particles, -1, dtype=int
        )  # Parent IDs (-1 indicates no parent)

        # Colony tracking
        self.colony_id: np.ndarray = np.full(
            n_particles, -1, dtype=int
        )  # Colony IDs (-1 indicates no colony)
        self.colony_role: np.ndarray = np.zeros(
            n_particles, dtype=int
        )  # Colony roles (0=none, 1=leader, 2=member)

        # Synergy network
        self.synergy_connections: np.ndarray = np.zeros(
            (n_particles, n_particles), dtype=bool
        )  # Synergy connection matrix

        # Adaptation metrics
        self.fitness_score: np.ndarray = np.zeros(
            n_particles, dtype=float
        )  # Individual fitness scores
        self.generation: np.ndarray = np.zeros(
            n_particles, dtype=int
        )  # Generation counter
        self.mutation_history: List[List[Tuple[str, float]]] = [
            [] for _ in range(n_particles)
        ]  # Track mutations

    def is_alive_mask(self) -> np.ndarray:
        """
        Compute a mask of alive cellular components based on various conditions:
        - Energy must be greater than 0.
        - If mass-based, mass must be greater than 0.
        - Age must be less than max_age.
        - Alive flag must be True.
        Returns:
        --------
        np.ndarray
            Boolean array indicating alive status of cellular components.
        """
        # Basic alive conditions: energy > 0, age < max_age, and alive flag is True
        mask = self.alive & (self.energy > 0.0) & (self.age < self.max_age)
        if self.mass_based and self.mass is not None:
            # Additional condition for mass-based types: mass > 0
            mask = mask & (self.mass > 0.0)
        return mask  # Return the computed alive mask

    def update_alive(self) -> None:
        """
        Update the alive status of cellular components based on current conditions.
        """
        self.alive = (
            self.is_alive_mask()
        )  # Update alive mask based on current conditions

    def age_components(self) -> None:
        """
        Increment the age of each cellular component and apply a minimal energy drain due to aging.
        """
        self.age += 1.0  # Increment age by 1 unit per frame
        self.energy = np.clip(
            self.energy, 0.0, None
        )  # Ensure energy doesn't go below 0

    def update_states(self) -> None:
        """
        Update the state of cellular components if needed.
        Currently a placeholder for future expansions (e.g., 'active', 'inactive').
        """
        pass  # No state arrays are stored; this can be expanded as needed

    def remove_dead(self, config: SimulationConfig) -> None:
        """
        Remove dead cellular components from the type to keep data arrays compact.
        Filters all component attributes based on the alive mask.
        Additionally, transfers energy from components dying of old age to their nearest three living neighbors.

        Parameters:
        -----------
        config : SimulationConfig
            Configuration parameters for the simulation, needed for predation_range.
        """
        # Validate array sizes first to prevent index mismatches
        array_size = self.x.size
        if not all(
            getattr(self, attr).size == array_size
            for attr in [
                "y",
                "vx",
                "vy",
                "energy",
                "alive",
                "age",
                "energy_efficiency",
                "speed_factor",
                "interaction_strength",
                "perception_range",
                "reproduction_rate",
                "synergy_affinity",
                "colony_factor",
                "drift_sensitivity",
                "species_id",
                "parent_id",
            ]
        ):
            # If arrays are mismatched, synchronize them to smallest size
            min_size = min(
                getattr(self, attr).size
                for attr in [
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "energy",
                    "alive",
                    "age",
                    "energy_efficiency",
                    "speed_factor",
                    "interaction_strength",
                    "perception_range",
                    "reproduction_rate",
                    "synergy_affinity",
                    "colony_factor",
                    "drift_sensitivity",
                    "species_id",
                    "parent_id",
                ]
            )
            for attr in [
                "x",
                "y",
                "vx",
                "vy",
                "energy",
                "alive",
                "age",
                "energy_efficiency",
                "speed_factor",
                "interaction_strength",
                "perception_range",
                "reproduction_rate",
                "synergy_affinity",
                "colony_factor",
                "drift_sensitivity",
                "species_id",
                "parent_id",
            ]:
                setattr(self, attr, getattr(self, attr)[:min_size])
            if self.mass_based and self.mass is not None:
                self.mass = self.mass[:min_size]

        # Get current alive status
        alive_mask = self.is_alive_mask()

        # Identify components dying of old age
        dead_due_to_age = (~alive_mask) & (self.age >= self.max_age)

        if np.any(dead_due_to_age):
            # Find indices of alive and dead-by-age components
            alive_indices = np.where(alive_mask)[0]
            dead_age_indices = np.where(dead_due_to_age)[0]

            if alive_indices.size > 0:
                # Build positions array for KD-Tree
                alive_positions = np.column_stack(
                    (self.x[alive_indices], self.y[alive_indices])
                )
                tree = cKDTree(alive_positions)

                # Process energy transfer in batches for better performance
                batch_size = 1000
                for i in range(0, len(dead_age_indices), batch_size):
                    batch_indices = dead_age_indices[i : i + batch_size]
                    dead_positions = np.column_stack(
                        (self.x[batch_indices], self.y[batch_indices])
                    )
                    dead_energies = self.energy[batch_indices]

                    # Find nearest neighbors for all dead components in batch
                    distances, neighbors = tree.query(
                        dead_positions, k=3, distance_upper_bound=config.predation_range
                    )

                    # Process energy transfer
                    valid_mask = distances < config.predation_range
                    for j, (dist_row, neighbor_row, dead_energy) in enumerate(
                        zip(distances, neighbors, dead_energies)
                    ):
                        valid = valid_mask[j]
                        if np.any(valid):
                            valid_neighbors = neighbor_row[valid]
                            energy_share = dead_energy / np.sum(valid)
                            self.energy[alive_indices[valid_neighbors]] += energy_share
                            self.energy[batch_indices[j]] = 0.0

        # Apply filtering to all arrays
        arrays_to_filter = [
            "x",
            "y",
            "vx",
            "vy",
            "energy",
            "alive",
            "age",
            "energy_efficiency",
            "speed_factor",
            "interaction_strength",
            "perception_range",
            "reproduction_rate",
            "synergy_affinity",
            "colony_factor",
            "drift_sensitivity",
            "species_id",
            "parent_id",
        ]

        for attr in arrays_to_filter:
            try:
                setattr(self, attr, getattr(self, attr)[alive_mask])
            except IndexError:
                # If error occurs, resize array to match alive_mask size
                current_array = getattr(self, attr)
                setattr(self, attr, current_array[: len(alive_mask)][alive_mask])

        # Handle mass array separately if it exists
        if self.mass_based and self.mass is not None:
            try:
                self.mass = self.mass[alive_mask]
            except IndexError:
                self.mass = self.mass[: len(alive_mask)][alive_mask]

    def add_component(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        energy: float,
        mass_val: Optional[float],
        energy_efficiency_val: float,
        speed_factor_val: float,
        interaction_strength_val: float,
        perception_range_val: float,
        reproduction_rate_val: float,
        synergy_affinity_val: float,
        colony_factor_val: float,
        drift_sensitivity_val: float,
        species_id_val: int,
        parent_id_val: int,
        max_age: float,
    ) -> None:
        """
        Add a new cellular component (e.g., offspring) to this cellular type.

        Parameters:
        -----------
        x : float
            X-coordinate of the new component.
        y : float
            Y-coordinate of the new component.
        vx : float
            X-component of the new component's velocity.
        vy : float
            Y-component of the new component's velocity.
        energy : float
            Initial energy of the new component.
        mass_val : Optional[float]
            Mass of the new component if type is mass-based; else None.
        energy_efficiency_val : float
            Energy efficiency trait of the new component.
        speed_factor_val : float
            Speed factor gene trait of the new component.
        interaction_strength_val : float
            Interaction strength gene trait of the new component.
        perception_range_val : float
            Perception range gene trait of the new component.
        reproduction_rate_val : float
            Reproduction rate gene trait of the new component.
        synergy_affinity_val : float
            Synergy affinity gene trait of the new component.
        colony_factor_val : float
            Colony formation gene trait of the new component.
        drift_sensitivity_val : float
            Drift sensitivity gene trait of the new component.
        species_id_val : int
            Species ID of the new component.
        parent_id_val : int
            Parent ID of the new component.
        max_age : float
            Maximum age for the new component.
        """
        # Append new component's attributes using NumPy's concatenate for efficiency
        self.x = np.concatenate((self.x, [x]))  # Append new x position
        self.y = np.concatenate((self.y, [y]))  # Append new y position
        self.vx = np.concatenate((self.vx, [vx]))  # Append new x velocity
        self.vy = np.concatenate((self.vy, [vy]))  # Append new y velocity
        self.energy = np.concatenate((self.energy, [energy]))  # Append new energy level
        self.alive = np.concatenate((self.alive, [True]))  # Set alive status to True
        self.age = np.concatenate(
            (self.age, [0.0])
        )  # Initialize age to 0 for new component
        self.energy_efficiency = np.concatenate(
            (self.energy_efficiency, [energy_efficiency_val])
        )  # Append energy efficiency
        self.speed_factor = np.concatenate(
            (self.speed_factor, [speed_factor_val])
        )  # Append speed factor
        self.interaction_strength = np.concatenate(
            (self.interaction_strength, [interaction_strength_val])
        )  # Append interaction strength
        self.perception_range = np.concatenate(
            (self.perception_range, [perception_range_val])
        )  # Append perception range
        self.reproduction_rate = np.concatenate(
            (self.reproduction_rate, [reproduction_rate_val])
        )  # Append reproduction rate
        self.synergy_affinity = np.concatenate(
            (self.synergy_affinity, [synergy_affinity_val])
        )  # Append synergy affinity
        self.colony_factor = np.concatenate(
            (self.colony_factor, [colony_factor_val])
        )  # Append colony factor
        self.drift_sensitivity = np.concatenate(
            (self.drift_sensitivity, [drift_sensitivity_val])
        )  # Append drift sensitivity
        self.species_id = np.concatenate(
            (self.species_id, [species_id_val])
        )  # Append species ID
        self.parent_id = np.concatenate(
            (self.parent_id, [parent_id_val])
        )  # Append parent ID

        if self.mass_based and self.mass is not None:
            if mass_val is None or mass_val <= 0.0:
                # Ensure mass is positive; assign a small random mass if invalid
                mass_val = max(0.1, abs(mass_val if mass_val is not None else 1.0))
            self.mass = np.concatenate((self.mass, [mass_val]))  # Append new mass value
