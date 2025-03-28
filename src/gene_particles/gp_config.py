"""
GeneParticles: Cellular Automata with Dynamic Gene Expression and Emergent Behaviors
------------------------------------------------------------------------------------
A particle simulation modeling cellular entities with genetic traits, adaptive behaviors,
emergent properties, and complex interaction networks.

Core Simulation Features:
1. Dynamic Gene Expression
    - Mutable traits affecting behavior and reproduction
    - Hierarchical gene structures with layered mutation strategies
    - Environment-responsive phenotype expression

2. Adaptive Population Management
    - Performance-aware optimization triggers
    - Fitness-weighted population culling
    - Resource-based population regulation

3. Evolutionary Mechanisms
    - Competitive resource dynamics
    - Multi-factor speciation events
    - Complete lineage tracking

4. Multi-Scale Interactions
    - Potential, gravitational, and synergistic forces
    - Species-dependent energy transfers
    - Emergent group behaviors (flocking, predation, colonies)

5. Vectorized Performance
    - NumPy-accelerated computation
    - Spatial partitioning optimization
    - Adaptive rendering systems

Technical Requirements:
- Python 3.8+
- NumPy >= 1.20.0
- Pygame >= 2.0.0
- SciPy >= 1.7.0

Usage: python geneparticles.py
Controls: ESC to exit
"""

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

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


@dataclass
class GeneticParamConfig:
    """
    Configuration for genetic parameters and mutation dynamics.

    Defines trait specifications, mutation rates, and constraint mechanisms
    to maintain biologically plausible simulation behaviors.

    Attributes:
         gene_traits: Registry of all genetic trait names
         gene_mutation_rate: Probability of trait mutation during reproduction
         gene_mutation_range: Bounds for mutation magnitude
         trait_definitions: Complete specifications for all genetic traits
         energy_efficiency_mutation_rate: Probability of energy efficiency mutation
         energy_efficiency_mutation_range: Bounds for energy efficiency mutation
    """

    gene_traits: List[str] = field(default_factory=list)
    gene_mutation_rate: float = 0.25
    gene_mutation_range: Range = (-0.2, 0.2)
    trait_definitions: Dict[str, TraitDefinition] = field(default_factory=dict)
    energy_efficiency_mutation_rate: float = 0.2
    energy_efficiency_mutation_range: Range = (-0.15, 0.3)

    def __post_init__(self) -> None:
        """Initialize trait definitions and validate configuration."""
        # Define core trait specifications
        self._initialize_trait_definitions()

        # Extract trait names from definitions
        self.gene_traits = list(self.trait_definitions.keys())

        # Validate configuration
        self._validate()

    def _initialize_trait_definitions(self) -> None:
        """
        Build trait specification registry with full trait metadata.

        Each trait receives a complete definition including valid ranges,
        behavioral category, and descriptive explanation.
        """
        self.trait_definitions = {
            "speed_factor": TraitDefinition(
                name="speed_factor",
                type=TraitType.MOVEMENT,
                range=(0.05, 4.0),
                description="Movement velocity multiplier",
                default=1.0,
            ),
            "interaction_strength": TraitDefinition(
                name="interaction_strength",
                type=TraitType.INTERACTION,
                range=(0.05, 4.0),
                description="Force interaction multiplier",
                default=1.0,
            ),
            "perception_range": TraitDefinition(
                name="perception_range",
                type=TraitType.PERCEPTION,
                range=(20.0, 400.0),
                description="Environmental sensing distance",
                default=100.0,
            ),
            "reproduction_rate": TraitDefinition(
                name="reproduction_rate",
                type=TraitType.REPRODUCTION,
                range=(0.02, 1.5),
                description="Probability of reproduction event",
                default=0.5,
            ),
            "synergy_affinity": TraitDefinition(
                name="synergy_affinity",
                type=TraitType.SOCIAL,
                range=(0.0, 3.0),
                description="Cooperative energy sharing strength",
                default=1.0,
            ),
            "colony_factor": TraitDefinition(
                name="colony_factor",
                type=TraitType.SOCIAL,
                range=(0.0, 2.0),
                description="Tendency to form collective groups",
                default=0.5,
            ),
            "drift_sensitivity": TraitDefinition(
                name="drift_sensitivity",
                type=TraitType.ADAPTATION,
                range=(0.0, 3.0),
                description="Responsiveness to evolutionary pressure",
                default=1.0,
            ),
        }

    def _validate(self) -> None:
        """
        Verify configuration integrity and parameter constraints.

        Raises:
              ValueError: If any parameter violates defined constraints
        """
        if not (0.0 <= self.gene_mutation_rate <= 1.0):
            raise ValueError("Gene mutation rate must be between 0.0 and 1.0")

        if self.gene_mutation_range[0] >= self.gene_mutation_range[1]:
            raise ValueError("Gene mutation range must have min < max")

        if not (0.0 <= self.energy_efficiency_mutation_rate <= 1.0):
            raise ValueError(
                "Energy efficiency mutation rate must be between 0.0 and 1.0"
            )

        if (
            self.energy_efficiency_mutation_range[0]
            >= self.energy_efficiency_mutation_range[1]
        ):
            raise ValueError("Energy efficiency mutation range must have min < max")

        # Verify all trait definitions exist
        for trait in self.gene_traits:
            if trait not in self.trait_definitions:
                raise ValueError(f"Missing trait definition for '{trait}'")

    def clamp_gene_values(
        self,
        speed_factor: FloatArray,
        interaction_strength: FloatArray,
        perception_range: FloatArray,
        reproduction_rate: FloatArray,
        synergy_affinity: FloatArray,
        colony_factor: FloatArray,
        drift_sensitivity: FloatArray,
    ) -> Tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """
        Constrain genetic trait arrays to their valid ranges.

        Prevents simulation instability by enforcing trait boundaries, acting as
        a genetic "immune system" against extreme mutations.

        Args:
              speed_factor: Movement velocity multipliers
              interaction_strength: Interaction force multipliers
              perception_range: Environmental sensing distances
              reproduction_rate: Reproduction probability values
              synergy_affinity: Energy sharing tendencies
              colony_factor: Group formation tendencies
              drift_sensitivity: Evolutionary responsiveness values

        Returns:
              Tuple of constrained arrays with identical shapes but values
              clamped to their respective trait definition ranges
        """
        return (
            np.clip(
                speed_factor,
                self.trait_definitions["speed_factor"].range[0],
                self.trait_definitions["speed_factor"].range[1],
            ),
            np.clip(
                interaction_strength,
                self.trait_definitions["interaction_strength"].range[0],
                self.trait_definitions["interaction_strength"].range[1],
            ),
            np.clip(
                perception_range,
                self.trait_definitions["perception_range"].range[0],
                self.trait_definitions["perception_range"].range[1],
            ),
            np.clip(
                reproduction_rate,
                self.trait_definitions["reproduction_rate"].range[0],
                self.trait_definitions["reproduction_rate"].range[1],
            ),
            np.clip(
                synergy_affinity,
                self.trait_definitions["synergy_affinity"].range[0],
                self.trait_definitions["synergy_affinity"].range[1],
            ),
            np.clip(
                colony_factor,
                self.trait_definitions["colony_factor"].range[0],
                self.trait_definitions["colony_factor"].range[1],
            ),
            np.clip(
                drift_sensitivity,
                self.trait_definitions["drift_sensitivity"].range[0],
                self.trait_definitions["drift_sensitivity"].range[1],
            ),
        )

    def get_range_for_trait(self, trait_name: str) -> Range:
        """
        Retrieve the valid range for a specific trait.

        Args:
              trait_name: Name of the genetic trait

        Returns:
              Tuple containing (min, max) valid values

        Raises:
              KeyError: If trait_name is not a valid genetic trait
        """
        if trait_name not in self.trait_definitions:
            raise KeyError(f"Unknown trait: {trait_name}")
        return self.trait_definitions[trait_name].range

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a serializable dictionary.

        Returns:
              Dictionary containing all configuration parameters with
              trait definitions converted to serializable format
        """
        result = {k: v for k, v in asdict(self).items() if k != "trait_definitions"}
        # Convert TraitDefinitions to dict
        result["trait_definitions"] = {
            name: asdict(definition)
            for name, definition in self.trait_definitions.items()
        }
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GeneticParamConfig":
        """
        Create a genetic parameter configuration from a dictionary.

        Args:
            config_dict: Dictionary containing genetic configuration parameters

        Returns:
            Fully initialized GeneticParamConfig instance

        Raises:
            ValueError: If dictionary contains invalid genetic configuration
        """
        instance = cls()

        # Handle trait definitions separately if present
        trait_defs = config_dict.pop("trait_definitions", {})
        if trait_defs:
            # Recreate TraitDefinition objects from dictionary representation
            instance.trait_definitions = {
                name: TraitDefinition(
                    name=def_dict["name"],
                    type=(
                        TraitType[def_dict["type"].name]
                        if isinstance(def_dict["type"], Enum)
                        else def_dict["type"]
                    ),
                    range=def_dict["range"],
                    description=def_dict["description"],
                    default=def_dict["default"],
                )
                for name, def_dict in trait_defs.items()
            }

        # Set all other attributes
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        return instance


class SimulationConfig:
    """
    Master configuration for the GeneParticles simulation ecosystem.

    Controls all simulation parameters including population dynamics,
    physical properties, energy systems, evolutionary mechanics,
    and emergent behavior tuning.

    Attributes:
         n_cell_types: Number of distinct cellular types
         particles_per_type: Initial particles per cellular type
         min_particles_per_type: Minimum particles threshold per type
         max_particles_per_type: Maximum particles allowed per type
         mass_range: Min/max mass values for mass-based particles
         base_velocity_scale: Base movement speed multiplier
         mass_based_fraction: Proportion of types using mass-based interactions
         interaction_strength_range: Min/max interaction force strengths
         max_frames: Maximum simulation frames (0 = infinite)
         initial_energy: Starting energy per particle
         friction: Environmental movement resistance (0-1)
         global_temperature: Random motion intensity
         predation_range: Maximum distance for predation interactions
         energy_transfer_factor: Efficiency of energy transfers
         mass_transfer: Whether mass transfers with energy
         max_age: Maximum particle lifespan (inf = immortal)
         evolution_interval: Frames between evolutionary updates
         synergy_range: Maximum distance for synergy interactions
         culling_fitness_weights: Weighting factors for particle fitness
         reproduction_energy_threshold: Minimum energy for reproduction
         reproduction_mutation_rate: Mutation probability during reproduction
         reproduction_offspring_energy_fraction: Energy fraction given to offspring
         alignment_strength: Strength of flocking alignment behavior
         cohesion_strength: Strength of flocking cohesion behavior
         separation_strength: Strength of flocking separation behavior
         cluster_radius: Radius for flocking neighbor detection
         particle_size: Visual size of particles
         energy_efficiency_range: Min/max energy efficiency values
         genetics: Nested genetic parameter configuration
         speciation_threshold: Genetic distance threshold for new species
         colony_formation_probability: Probability of colony formation
         colony_radius: Maximum colony size
         colony_cohesion_strength: Force pulling colony members together
         synergy_evolution_rate: Rate of synergy relationship changes
         complexity_factor: Multiplier for behavior complexity
         structural_complexity_weight: Emphasis on emergent structures
    """

    def __init__(self) -> None:
        """Initialize simulation configuration with default parameters."""
        # Population parameters
        self.n_cell_types: int = 20
        self.particles_per_type: int = 50
        self.min_particles_per_type: int = 50
        self.max_particles_per_type: int = 300

        # Physical parameters
        self.mass_range: Range = (0.2, 15.0)
        self.base_velocity_scale: float = 0.8
        self.mass_based_fraction: float = 0.7
        self.interaction_strength_range: Range = (-3.0, 3.0)
        self.friction: float = 0.25
        self.global_temperature: float = 0.05  # Random motion intensity

        # Energy system parameters
        self.initial_energy: float = 150.0
        self.predation_range: float = 75.0
        self.energy_transfer_factor: float = 0.7
        self.mass_transfer: bool = True
        self.energy_efficiency_range: Range = (-0.4, 3.0)
        self.max_energy: float = 300.0  # Maximum energy for particles

        # Lifecycle parameters
        self.max_frames: int = 0  # 0 = infinite
        self.max_age: float = np.inf
        self.evolution_interval: int = 20000
        self.synergy_range: float = 200.0

        # Reproduction parameters
        self.reproduction_energy_threshold: float = 180.0
        self.reproduction_mutation_rate: float = 0.3
        self.reproduction_offspring_energy_fraction: float = 0.5

        # Flocking behavior parameters
        self.alignment_strength: float = 0.1
        self.cohesion_strength: float = 0.8
        self.separation_strength: float = 0.3
        self.cluster_radius: float = 10.0

        # Visualization parameters
        self.particle_size: float = 3.0

        # Fitness and selection parameters
        self.culling_fitness_weights: Dict[str, float] = {
            "energy_weight": 0.6,
            "age_weight": 0.8,
            "speed_factor_weight": 0.7,
            "interaction_strength_weight": 0.7,
            "synergy_affinity_weight": 0.8,
            "colony_factor_weight": 0.9,
            "drift_sensitivity_weight": 0.6,
        }

        # Emergence parameters
        self.speciation_threshold: float = 0.8
        self.colony_formation_probability: float = 0.4
        self.colony_radius: float = 250.0
        self.colony_cohesion_strength: float = 0.8
        self.synergy_evolution_rate: float = 0.08
        self.complexity_factor: float = 2.0
        self.structural_complexity_weight: float = 0.9

        # Initialize genetic parameters
        self.genetics = GeneticParamConfig()

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """
        Verify all configuration parameters for simulation stability.

        Performs comprehensive validation to ensure parameters are within
        acceptable ranges, preventing crashes or unexpected behaviors.

        Raises:
              ValueError: If any parameter violates its constraints
        """
        # Population validation
        if self.n_cell_types <= 0:
            raise ValueError("Number of cell types must be greater than 0")

        if self.particles_per_type <= 0:
            raise ValueError("Particles per type must be greater than 0")

        if self.min_particles_per_type <= 0:
            raise ValueError("Minimum particles per type must be greater than 0")

        if self.max_particles_per_type < self.min_particles_per_type:
            raise ValueError("Maximum particles must be ≥ minimum particles")

        # Physical properties validation
        if self.mass_range[0] <= 0:
            raise ValueError("Minimum mass must be positive")

        if self.mass_range[0] >= self.mass_range[1]:
            raise ValueError("Mass range must have min < max")

        if self.base_velocity_scale <= 0:
            raise ValueError("Base velocity scale must be positive")

        if not (0.0 <= self.mass_based_fraction <= 1.0):
            raise ValueError("Mass-based fraction must be between 0.0 and 1.0")

        if self.interaction_strength_range[0] >= self.interaction_strength_range[1]:
            raise ValueError("Interaction strength range must have min < max")

        # Energy system validation
        if self.initial_energy <= 0:
            raise ValueError("Initial energy must be positive")

        if not (0.0 <= self.friction <= 1.0):
            raise ValueError("Friction must be between 0.0 and 1.0")

        if self.global_temperature < 0:
            raise ValueError("Global temperature must be non-negative")

        if self.predation_range <= 0:
            raise ValueError("Predation range must be positive")

        if not (0.0 <= self.energy_transfer_factor <= 1.0):
            raise ValueError("Energy transfer factor must be between 0.0 and 1.0")

        if self.energy_efficiency_range[0] >= self.energy_efficiency_range[1]:
            raise ValueError("Energy efficiency range must have min < max")

        # Lifecycle validation
        if self.max_frames < 0:
            raise ValueError("Maximum frames must be non-negative")

        if self.synergy_range <= 0:
            raise ValueError("Synergy range must be positive")

        # Reproduction validation
        if self.reproduction_energy_threshold <= 0:
            raise ValueError("Reproduction energy threshold must be positive")

        if not (0.0 <= self.reproduction_mutation_rate <= 1.0):
            raise ValueError("Reproduction mutation rate must be between 0.0 and 1.0")

        if not (0.0 <= self.reproduction_offspring_energy_fraction <= 1.0):
            raise ValueError(
                "Reproduction offspring energy fraction must be between 0.0 and 1.0"
            )

        # Flocking validation
        if self.cluster_radius <= 0:
            raise ValueError("Cluster radius must be positive")

        # Visualization validation
        if self.particle_size <= 0:
            raise ValueError("Particle size must be positive")

        # Emergence validation
        if self.speciation_threshold <= 0:
            raise ValueError("Speciation threshold must be positive")

        if not (0.0 <= self.colony_formation_probability <= 1.0):
            raise ValueError("Colony formation probability must be between 0.0 and 1.0")

        if self.colony_radius <= 0:
            raise ValueError("Colony radius must be positive")

        if not (0.0 <= self.colony_cohesion_strength <= 1.0):
            raise ValueError("Colony cohesion strength must be between 0.0 and 1.0")

        if not (0.0 <= self.synergy_evolution_rate <= 1.0):
            raise ValueError("Synergy evolution rate must be between 0.0 and 1.0")

        if self.complexity_factor <= 0:
            raise ValueError("Complexity factor must be positive")

        if not (0.0 <= self.structural_complexity_weight <= 1.0):
            raise ValueError("Structural complexity weight must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire configuration to a flat dictionary representation.

        Creates a serializable representation of all parameters including
        nested configurations for logging, persistence, or visualization.

        Returns:
              Dictionary containing all configuration parameters and
              nested genetics configuration as nested dictionary
        """
        # Start with all attributes except genetics
        config_dict = {k: v for k, v in self.__dict__.items() if k != "genetics"}

        # Add genetics as a nested dictionary
        config_dict["genetics"] = self.genetics.to_dict()

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """
        Create a configuration instance from a dictionary.

        Args:
              config_dict: Dictionary containing configuration parameters

        Returns:
              Fully initialized SimulationConfig instance

        Raises:
              ValueError: If dictionary contains invalid configuration
        """
        instance = cls()

        # Extract genetics config if present and properly initialize it
        genetics_dict = config_dict.pop("genetics", None)
        if genetics_dict:
            instance.genetics = GeneticParamConfig.from_dict(genetics_dict)

        # Set all other attributes
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        # Validate after setting all values
        instance._validate()
        return instance
