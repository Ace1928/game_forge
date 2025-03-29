from typing import TYPE_CHECKING, Dict, List

import numpy as np

# Use TYPE_CHECKING for circular imports
if TYPE_CHECKING:
    from game_forge.src.gene_particles.gp_config import SimulationConfig
    from game_forge.src.gene_particles.gp_types import CellularTypeData

from game_forge.src.gene_particles.gp_types import FloatArray, GeneData
from game_forge.src.gene_particles.gp_utility import mutate_trait


def apply_movement_gene(
    particle: "CellularTypeData", gene_data: GeneData, env: "SimulationConfig"
) -> None:
    """Apply genes controlling movement behavior.

    Adjusts particle velocity based on genetic parameters, environmental
    friction, and stochastic factors through vectorized operations.

    Args:
        particle: Cellular type data of the particle
        gene_data: Movement parameters [speed_modifier, randomness, direction_bias]
        env: Environmental configuration parameters
    """
    # Extract gene parameters with defaults for missing values
    speed_modifier = gene_data[0] if len(gene_data) > 0 else 1.0
    randomness = gene_data[1] if len(gene_data) > 1 else 0.1
    direction_bias = gene_data[2] if len(gene_data) > 2 else 0.0

    # Clamp values to reasonable ranges
    speed_modifier = np.clip(speed_modifier, 0.1, 3.0)
    randomness = np.clip(randomness, 0.0, 1.0)
    direction_bias = np.clip(direction_bias, -1.0, 1.0)

    # Apply movement modifications vectorized
    friction_factor = 1.0 - env.friction  # Convert friction to retention factor

    # X-velocity: retained momentum + stochastic component + directional bias
    particle.vx = (
        particle.vx * friction_factor * speed_modifier
        + randomness * np.random.uniform(-1, 1, size=particle.vx.size)
        + direction_bias
    )

    # Y-velocity: retained momentum + stochastic component + directional bias
    particle.vy = (
        particle.vy * friction_factor * speed_modifier
        + randomness * np.random.uniform(-1, 1, size=particle.vy.size)
        + direction_bias
    )

    # Apply energy cost for movement (proportional to velocity magnitude)
    energy_cost = np.sqrt(np.power(particle.vx, 2) + np.power(particle.vy, 2)) * 0.01
    particle.energy = np.maximum(0.0, particle.energy - energy_cost)


def apply_interaction_gene(
    particle: "CellularTypeData",
    others: List["CellularTypeData"],
    gene_data: GeneData,
    env: "SimulationConfig",
) -> None:
    """Apply interaction-related behavior based on proximity.

    Calculates forces between particles based on distance and genetic
    attraction strength, simulating social behaviors like flocking,
    avoidance, or cooperation through vectorized force calculations.

    Args:
        particle: Cellular type data of the particle
        others: List of other cellular types for interaction calculations
        gene_data: Interaction parameters [attraction_strength, interaction_radius]
        env: Environmental configuration parameters
    """
    # Extract gene parameters with defaults for missing values
    attraction_strength = gene_data[0] if len(gene_data) > 0 else 0.5
    interaction_radius = gene_data[1] if len(gene_data) > 1 else 100.0

    # Clamp values to reasonable ranges
    attraction_strength = np.clip(attraction_strength, -2.0, 2.0)
    interaction_radius = np.clip(interaction_radius, 10.0, 300.0)

    for other in others:
        if other == particle:
            continue  # Skip self-interaction

        # Calculate distances and angles vectorized
        dx = other.x - particle.x[:, np.newaxis]  # Broadcasting for all combinations
        dy = other.y - particle.y[:, np.newaxis]
        distances = np.sqrt(np.power(dx, 2) + np.power(dy, 2))  # Euclidean distance

        # Create interaction mask - only interact within radius
        interact_mask = (distances > 0.0) & (distances < interaction_radius)

        if not np.any(interact_mask):
            continue  # Skip if no particles are within interaction range

        # Calculate normalized direction vectors (safely handle division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            dx_norm = np.where(distances > 0, dx / distances, 0)
            dy_norm = np.where(distances > 0, dy / distances, 0)

        # Calculate force magnitudes with linear distance falloff
        force_magnitudes = attraction_strength * (
            1.0 - distances / interaction_radius
        )  # Forces strongest at close range

        # Apply forces to update velocity vectors
        particle.vx += np.sum(dx_norm * force_magnitudes * interact_mask, axis=1)
        particle.vy += np.sum(dy_norm * force_magnitudes * interact_mask, axis=1)

        # Apply small energy cost for interaction (communication cost)
        particle.energy -= 0.01 * np.sum(interact_mask, axis=1)
        particle.energy = np.maximum(0.0, particle.energy)  # Prevent negative energy


def apply_energy_gene(
    particle: "CellularTypeData", gene_data: GeneData, env: "SimulationConfig"
) -> None:
    """Regulate energy dynamics based on genetic and environmental factors.

    Controls passive energy gain, energy efficiency, and age-based energy
    decay to simulate metabolic processes through vectorized operations.

    Args:
        particle: Cellular type data of the particle
        gene_data: Energy parameters [passive_gain, feeding_efficiency, predation_efficiency]
        env: Environmental configuration parameters
    """
    # Extract gene parameters with defaults for missing values
    passive_gain = gene_data[0] if len(gene_data) > 0 else 0.1
    feeding_efficiency = gene_data[1] if len(gene_data) > 1 else 0.5
    predation_efficiency = gene_data[2] if len(gene_data) > 2 else 0.3

    # Clamp values to reasonable ranges
    passive_gain = np.clip(passive_gain, 0.0, 0.5)  # Max 0.5 energy gain per tick
    feeding_efficiency = np.clip(feeding_efficiency, 0.1, 1.0)  # 10-100% efficient
    predation_efficiency = np.clip(predation_efficiency, 0.1, 1.0)  # 10-100% efficient

    # Calculate base energy gain modified by efficiency traits
    base_gain = passive_gain * particle.energy_efficiency

    # Apply environmental modifiers (e.g., day/night cycle, temperature)
    # This would be expanded to use env parameters in a more complex simulation
    env_modifier = 1.0

    # Calculate total energy gain with feeding efficiency factor
    energy_gain = base_gain * env_modifier * feeding_efficiency

    # Apply energy changes vectorized
    particle.energy += energy_gain

    # Apply energy decay based on age (senescence)
    age_factor = np.clip(
        particle.age / particle.max_age, 0.0, 1.0
    )  # 0 for young, 1 for old
    energy_decay = 0.01 * age_factor  # Older particles lose energy faster
    particle.energy = np.maximum(0.0, particle.energy - energy_decay)

    # Clamp energy to maximum value
    particle.energy = np.minimum(particle.energy, 200.0)  # Energy capacity limit


def apply_growth_gene(
    particle: "CellularTypeData", gene_data: GeneData, env: "SimulationConfig"
) -> None:
    """Apply growth gene to energy levels.

    Modifies energy levels based on a growth factor, ensuring values remain
    within particle-specific physiological bounds through vectorized clamping.

    Args:
        particle: Cellular type data to apply growth to
        gene_data: Growth parameters [growth_rate, adult_size, maturity_age]
        env: Environmental configuration parameters
    """
    # Extract growth factor with default fallback
    growth_factor = gene_data[0] if len(gene_data) > 0 else 0.1

    # Apply growth with min/max energy bounds from particle's configuration
    min_energy = particle.min_energy  # Use particle's min_energy
    max_energy = particle.max_energy  # Use particle's max_energy

    # Apply growth to energy levels
    particle.energy = np.clip(particle.energy * growth_factor, min_energy, max_energy)


def apply_reproduction_gene(
    particle: "CellularTypeData",
    others: List[
        "CellularTypeData"
    ],  # pylint: disable=unused-argument  # Kept for future sexual reproduction implementation
    gene_data: GeneData,
    env: "SimulationConfig",
) -> None:
    """Handle sexual and asexual reproduction mechanics.

    Controls particle reproduction based on energy thresholds, creates
    offspring with inherited traits and mutations, and manages speciation
    through genetic distance calculations and trait inheritance patterns.

    Args:
        particle: Cellular type data of the particle
        others: List of other cellular types for potential sexual reproduction
        gene_data: Reproduction parameters [sexual_threshold, asexual_threshold,
                   reproduction_cost, cooldown_time]
        env: Environmental configuration with genetics parameters
    """
    # Extract gene parameters with defaults for missing values
    sexual_threshold = gene_data[0] if len(gene_data) > 0 else 150.0
    asexual_threshold = gene_data[1] if len(gene_data) > 1 else 100.0
    reproduction_cost = gene_data[2] if len(gene_data) > 2 else 50.0
    cooldown_time = gene_data[3] if len(gene_data) > 3 else 30.0

    # Clamp values to reasonable ranges
    sexual_threshold = np.clip(sexual_threshold, 100.0, 200.0)
    asexual_threshold = np.clip(asexual_threshold, 50.0, 150.0)
    reproduction_cost = np.clip(reproduction_cost, 25.0, 100.0)
    cooldown_time = np.clip(cooldown_time, 10.0, 100.0)

    # Check reproduction conditions (energy > threshold, mature age, alive)
    can_reproduce = (
        (particle.energy > asexual_threshold)
        & (particle.age > cooldown_time)
        & particle.alive
    )

    if not np.any(can_reproduce):
        return  # No particles ready to reproduce

    # Get indices of particles that can reproduce
    reproduce_indices = np.where(can_reproduce)[0]

    for idx in reproduce_indices:
        # Deduct energy cost (reproduction is metabolically expensive)
        particle.energy[idx] -= reproduction_cost

        # Create offspring with inherited traits and mutations
        mutation_rate = env.genetics.gene_mutation_rate
        mutation_range = env.genetics.gene_mutation_range

        # Define trait max ranges for mutation boundaries
        max_ranges = {
            "energy_efficiency": 1.0,
            "speed_factor": 2.0,
            "interaction_strength": 2.0,
            "perception_range": 300.0,
            "reproduction_rate": 1.0,
            "synergy_affinity": 1.0,
            "colony_factor": 1.0,
            "drift_sensitivity": 1.0,
        }

        # Apply mutations to all traits using proper types
        # Create arrays for single-value traits to satisfy typing requirements
        offspring_traits: Dict[str, FloatArray] = {
            "energy_efficiency": mutate_trait(
                np.array([particle.energy_efficiency[idx]]),  # Convert to FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                mutation_range[1],  # max_range as float
            ),
            "speed_factor": mutate_trait(
                np.array([particle.speed_factor[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["speed_factor"],  # max_range as float
            ),
            "interaction_strength": mutate_trait(
                np.array([particle.interaction_strength[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["interaction_strength"],  # max_range as float
            ),
            "perception_range": mutate_trait(
                np.array([particle.perception_range[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["perception_range"],  # max_range as float
            ),
            "reproduction_rate": mutate_trait(
                np.array([particle.reproduction_rate[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["reproduction_rate"],  # max_range as float
            ),
            "synergy_affinity": mutate_trait(
                np.array([particle.synergy_affinity[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["synergy_affinity"],  # max_range as float
            ),
            "colony_factor": mutate_trait(
                np.array([particle.colony_factor[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["colony_factor"],  # max_range as float
            ),
            "drift_sensitivity": mutate_trait(
                np.array([particle.drift_sensitivity[idx]]),  # FloatArray
                np.array([mutation_rate > np.random.random()], dtype=bool),  # BoolArray
                mutation_range[0],  # min_range as float
                max_ranges["drift_sensitivity"],  # max_range as float
            ),
        }

        # Calculate genetic distance for speciation (multidimensional trait distance)
        genetic_distance = np.sqrt(
            np.power(offspring_traits["speed_factor"] - particle.speed_factor[idx], 2)
            + np.power(
                offspring_traits["interaction_strength"]
                - particle.interaction_strength[idx],
                2,
            )
            + np.power(
                offspring_traits["perception_range"] - particle.perception_range[idx],
                2,
            )
            + np.power(
                offspring_traits["reproduction_rate"] - particle.reproduction_rate[idx],
                2,
            )
            + np.power(
                offspring_traits["synergy_affinity"] - particle.synergy_affinity[idx],
                2,
            )
            + np.power(
                offspring_traits["colony_factor"] - particle.colony_factor[idx], 2
            )
            + np.power(
                offspring_traits["drift_sensitivity"] - particle.drift_sensitivity[idx],
                2,
            )
        )

        # Determine species ID (speciation occurs when genetic distance exceeds threshold)
        if genetic_distance > env.speciation_threshold:
            species_id_val = int(np.max(particle.species_id)) + 1  # New species
        else:
            species_id_val = particle.species_id[idx]  # Same species as parent

        # Add offspring to particle data (slightly offset from parent position)
        particle.add_component(
            x=float(particle.x[idx] + np.random.uniform(-5, 5)),
            y=float(particle.y[idx] + np.random.uniform(-5, 5)),
            vx=float(particle.vx[idx] * np.random.uniform(0.9, 1.1)),
            vy=float(particle.vy[idx] * np.random.uniform(0.9, 1.1)),
            energy=float(particle.energy[idx] * 0.5),
            mass_val=(
                float(particle.mass[idx])
                if particle.mass_based and particle.mass is not None
                else None
            ),
            # Extract scalar float values from numpy arrays with array[0]
            energy_efficiency_val=float(offspring_traits["energy_efficiency"][0]),
            speed_factor_val=float(offspring_traits["speed_factor"][0]),
            interaction_strength_val=float(offspring_traits["interaction_strength"][0]),
            perception_range_val=float(offspring_traits["perception_range"][0]),
            reproduction_rate_val=float(offspring_traits["reproduction_rate"][0]),
            synergy_affinity_val=float(offspring_traits["synergy_affinity"][0]),
            colony_factor_val=float(offspring_traits["colony_factor"][0]),
            drift_sensitivity_val=float(offspring_traits["drift_sensitivity"][0]),
            species_id_val=int(species_id_val),
            parent_id_val=int(particle.type_id),
            max_age=float(particle.max_age),
        )


def apply_predation_gene(
    particle: "CellularTypeData",
    others: List["CellularTypeData"],
    gene_data: GeneData,
    env: "SimulationConfig",
) -> None:
    """Apply predation behaviors based on attack power and energy gain.

    Controls predatory interactions between particles, including target selection,
    attack success probability, energy transfer mechanics, and ecological
    population dynamics.

    Args:
        particle: Cellular type data of the predator particle
        others: List of other cellular types for potential predation targets
        gene_data: Predation parameters [attack_power, energy_gain]
        env: Environmental configuration parameters
    """
    # This method would implement predation logic
    # For now, it's a placeholder that will be implemented later
    pass
