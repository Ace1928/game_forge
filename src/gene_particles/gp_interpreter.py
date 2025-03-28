from gp_config import SimulationConfig
from gp_types import CellularTypeData

###############################################################
# Genetic Interpreter Class
###############################################################


class GeneticInterpreter:
    """
    Decodes and interprets the genetic sequence of particles.
    Genes define behavior, traits, reproduction, and interactions.
    Each gene includes redundant components and multiple mutable parts.
    """

    def __init__(self, gene_sequence: Optional[List[List[Any]]] = None):
        """
        Initialize the GeneticInterpreter with a gene sequence.

        Parameters:
        -----------
        gene_sequence : Optional[List[List[Any]]], default=None
            Encoded genetic data (list of symbolic gene arrays). If None, uses default sequence.
        """
        # Default gene sequence if none provided
        self.default_sequence = [
            [
                "start_movement",
                1.0,
                0.1,
                0.0,
            ],  # [speed_modifier, randomness, direction_bias]
            [
                "start_interaction",
                0.5,
                100.0,
            ],  # [attraction_strength, interaction_radius]
            [
                "start_energy",
                0.1,
                0.5,
                0.3,
            ],  # [passive_gain, feeding_efficiency, predation_efficiency]
            [
                "start_reproduction",
                150.0,
                100.0,
                50.0,
                30.0,
            ],  # [sexual_threshold, asexual_threshold, reproduction_cost, cooldown_time]
            [
                "start_growth",
                0.1,
                2.0,
                100.0,
            ],  # [growth_rate, adult_size, maturity_age]
            ["start_predation", 10.0, 5.0],  # [attack_power, energy_gain]
        ]
        self.gene_sequence = (
            gene_sequence if gene_sequence is not None else self.default_sequence
        )

    def decode(
        self,
        particle: CellularTypeData,
        others: List[CellularTypeData],
        env: SimulationConfig,
    ) -> None:
        """
        Decodes the genetic sequence to influence particle behavior.
        Each gene's symbolic structure is mapped to corresponding traits or actions.

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle to decode.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Iterate through each gene in the gene sequence
        for gene in self.gene_sequence:
            if not gene or len(gene) < 2:
                continue  # Skip invalid genes

            gene_type = gene[0]
            gene_data = gene[1:]

            try:
                if gene_type == "start_movement":
                    self.apply_movement_gene(particle, gene_data, env)
                elif gene_type == "start_interaction":
                    self.apply_interaction_gene(particle, others, gene_data, env)
                elif gene_type == "start_energy":
                    self.apply_energy_gene(particle, gene_data, env)
                elif gene_type == "start_reproduction":
                    self.apply_reproduction_gene(particle, others, gene_data, env)
                elif gene_type == "start_growth":
                    self.apply_growth_gene(particle, gene_data)
                elif gene_type == "start_predation":
                    self.apply_predation_gene(particle, others, gene_data, env)
            except Exception as e:
                print(f"Error processing gene {gene_type}: {str(e)}")

    def apply_movement_gene(
        self, particle: CellularTypeData, gene_data: List[Any], env: SimulationConfig
    ) -> None:
        """
        Apply genes controlling movement behavior.
        gene_data: [speed_modifier, randomness, direction_bias]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        gene_data : List[Any]
            Data for the movement gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        speed_modifier = gene_data[0] if len(gene_data) > 0 else 1.0
        randomness = gene_data[1] if len(gene_data) > 1 else 0.1
        direction_bias = gene_data[2] if len(gene_data) > 2 else 0.0

        # Clamp values to reasonable ranges
        speed_modifier = np.clip(speed_modifier, 0.1, 3.0)
        randomness = np.clip(randomness, 0.0, 1.0)
        direction_bias = np.clip(direction_bias, -1.0, 1.0)

        # Apply movement modifications vectorized
        friction_factor = 1.0 - env.friction
        particle.vx = (
            particle.vx * friction_factor * speed_modifier
            + randomness * np.random.uniform(-1, 1, size=particle.vx.size)
            + direction_bias
        )
        particle.vy = (
            particle.vy * friction_factor * speed_modifier
            + randomness * np.random.uniform(-1, 1, size=particle.vy.size)
            + direction_bias
        )

        # Apply energy cost for movement
        energy_cost = np.sqrt(particle.vx**2 + particle.vy**2) * 0.01
        particle.energy = np.maximum(0.0, particle.energy - energy_cost)

    def apply_interaction_gene(
        self,
        particle: CellularTypeData,
        others: List[CellularTypeData],
        gene_data: List[Any],
        env: SimulationConfig,
    ) -> None:
        """
        Apply interaction-related behavior based on proximity.
        gene_data: [attraction_strength, interaction_radius]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        gene_data : List[Any]
            Data for the interaction gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        attraction_strength = gene_data[0] if len(gene_data) > 0 else 0.5
        interaction_radius = gene_data[1] if len(gene_data) > 1 else 100.0

        # Clamp values to reasonable ranges
        attraction_strength = np.clip(attraction_strength, -2.0, 2.0)
        interaction_radius = np.clip(interaction_radius, 10.0, 300.0)

        for other in others:
            if other == particle:
                continue

            # Calculate distances and angles vectorized
            dx = other.x - particle.x[:, np.newaxis]
            dy = other.y - particle.y[:, np.newaxis]
            distances = np.sqrt(dx**2 + dy**2)

            # Create interaction mask
            interact_mask = (distances > 0.0) & (distances < interaction_radius)

            if not np.any(interact_mask):
                continue

            # Calculate normalized direction vectors
            with np.errstate(divide="ignore", invalid="ignore"):
                dx_norm = np.where(distances > 0, dx / distances, 0)
                dy_norm = np.where(distances > 0, dy / distances, 0)

            # Calculate force magnitudes with distance falloff
            force_magnitudes = attraction_strength * (
                1.0 - distances / interaction_radius
            )

            # Apply forces
            particle.vx += np.sum(dx_norm * force_magnitudes * interact_mask, axis=1)
            particle.vy += np.sum(dy_norm * force_magnitudes * interact_mask, axis=1)

            # Apply small energy cost for interaction
            particle.energy -= 0.01 * np.sum(interact_mask, axis=1)
            particle.energy = np.maximum(0.0, particle.energy)

    def apply_energy_gene(
        self, particle: CellularTypeData, gene_data: List[Any], env: SimulationConfig
    ) -> None:
        """
        Regulates energy dynamics based on genetic and environmental modifiers.
        gene_data: [passive_gain, feeding_efficiency, predation_efficiency]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        gene_data : List[Any]
            Data for the energy gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        passive_gain = gene_data[0] if len(gene_data) > 0 else 0.1
        feeding_efficiency = gene_data[1] if len(gene_data) > 1 else 0.5
        predation_efficiency = gene_data[2] if len(gene_data) > 2 else 0.3

        # Clamp values to reasonable ranges
        passive_gain = np.clip(passive_gain, 0.0, 0.5)
        feeding_efficiency = np.clip(feeding_efficiency, 0.1, 1.0)
        predation_efficiency = np.clip(predation_efficiency, 0.1, 1.0)

        # Calculate base energy gain modified by efficiency traits
        base_gain = passive_gain * particle.energy_efficiency

        # Apply environmental modifiers (e.g., day/night cycle, temperature)
        env_modifier = 1.0  # Could be modified based on env parameters

        # Calculate total energy gain
        energy_gain = base_gain * env_modifier * feeding_efficiency

        # Apply energy changes vectorized
        particle.energy += energy_gain

        # Apply energy decay based on age
        age_factor = np.clip(particle.age / particle.max_age, 0.0, 1.0)
        energy_decay = 0.01 * age_factor
        particle.energy = np.maximum(0.0, particle.energy - energy_decay)

        # Clamp energy to maximum value
        particle.energy = np.minimum(particle.energy, 200.0)

    def apply_reproduction_gene(
        self,
        particle: CellularTypeData,
        others: List[CellularTypeData],
        gene_data: List[Any],
        env: SimulationConfig,
    ) -> None:
        """
        Handles sexual and asexual reproduction.
        gene_data: [sexual_threshold, asexual_threshold, reproduction_cost, cooldown_time]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        gene_data : List[Any]
            Data for the reproduction gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        sexual_threshold = gene_data[0] if len(gene_data) > 0 else 150.0
        asexual_threshold = gene_data[1] if len(gene_data) > 1 else 100.0
        reproduction_cost = gene_data[2] if len(gene_data) > 2 else 50.0
        cooldown_time = gene_data[3] if len(gene_data) > 3 else 30.0

        # Clamp values to reasonable ranges
        sexual_threshold = np.clip(sexual_threshold, 100.0, 200.0)
        asexual_threshold = np.clip(asexual_threshold, 50.0, 150.0)
        reproduction_cost = np.clip(reproduction_cost, 25.0, 100.0)
        cooldown_time = np.clip(cooldown_time, 10.0, 100.0)

        # Check reproduction conditions
        can_reproduce = (
            (particle.energy > asexual_threshold)
            & (particle.age > cooldown_time)
            & particle.alive
        )

        if not np.any(can_reproduce):
            return

        # Get indices of particles that can reproduce
        reproduce_indices = np.where(can_reproduce)[0]

        for idx in reproduce_indices:
            # Deduct energy cost
            particle.energy[idx] -= reproduction_cost

            # Create offspring with inherited traits and mutations
            mutation_rate = env.genetics.gene_mutation_rate
            mutation_range = env.genetics.gene_mutation_range

            # Apply mutations to all traits
            offspring_traits = {
                "energy_efficiency": self._mutate_trait(
                    particle.energy_efficiency[idx], mutation_rate, mutation_range
                ),
                "speed_factor": self._mutate_trait(
                    particle.speed_factor[idx], mutation_rate, mutation_range
                ),
                "interaction_strength": self._mutate_trait(
                    particle.interaction_strength[idx], mutation_rate, mutation_range
                ),
                "perception_range": self._mutate_trait(
                    particle.perception_range[idx], mutation_rate, mutation_range
                ),
                "reproduction_rate": self._mutate_trait(
                    particle.reproduction_rate[idx], mutation_rate, mutation_range
                ),
                "synergy_affinity": self._mutate_trait(
                    particle.synergy_affinity[idx], mutation_rate, mutation_range
                ),
                "colony_factor": self._mutate_trait(
                    particle.colony_factor[idx], mutation_rate, mutation_range
                ),
                "drift_sensitivity": self._mutate_trait(
                    particle.drift_sensitivity[idx], mutation_rate, mutation_range
                ),
            }

            # Calculate genetic distance for speciation
            genetic_distance = np.sqrt(
                (offspring_traits["speed_factor"] - particle.speed_factor[idx]) ** 2
                + (
                    offspring_traits["interaction_strength"]
                    - particle.interaction_strength[idx]
                )
                ** 2
                + (
                    offspring_traits["perception_range"]
                    - particle.perception_range[idx]
                )
                ** 2
                + (
                    offspring_traits["reproduction_rate"]
                    - particle.reproduction_rate[idx]
                )
                ** 2
                + (
                    offspring_traits["synergy_affinity"]
                    - particle.synergy_affinity[idx]
                )
                ** 2
                + (offspring_traits["colony_factor"] - particle.colony_factor[idx]) ** 2
                + (
                    offspring_traits["drift_sensitivity"]
                    - particle.drift_sensitivity[idx]
                )
                ** 2
            )

            # Determine species ID
            if genetic_distance > env.speciation_threshold:
                species_id_val = int(np.max(particle.species_id)) + 1
            else:
                species_id_val = particle.species_id[idx]

            # Add offspring to particle data
            particle.add_component(
                x=particle.x[idx] + np.random.uniform(-5, 5),
                y=particle.y[idx] + np.random.uniform(-5, 5),
                vx=particle.vx[idx] * np.random.uniform(0.9, 1.1),
                vy=particle.vy[idx] * np.random.uniform(0.9, 1.1),
                energy=particle.energy[idx] * 0.5,
                mass_val=particle.mass[idx] if particle.mass_based else None,
                energy_efficiency_val=offspring_traits["energy_efficiency"],
                speed_factor_val=offspring_traits["speed_factor"],
                interaction_strength_val=offspring_traits["interaction_strength"][idx],
                perception_range_val=offspring_traits["perception_range"][idx],
                reproduction_rate_val=offspring_traits["reproduction_rate"][idx],
                synergy_affinity_val=offspring_traits["synergy_affinity"][idx],
                colony_factor_val=offspring_traits["colony_factor"][idx],
                drift_sensitivity_val=offspring_traits["drift_sensitivity"][idx],
                species_id_val=species_id_val,
                parent_id_val=particle.type_id,
                max_age=particle.max_age,
            )

    def _mutate_trait(
        self,
        base_value: float,
        mutation_rate: float,
        mutation_range: Tuple[float, float],
    ) -> float:
        """Helper method to mutate a trait value with given parameters."""
        if np.random.random() < mutation_rate:
            mutation = np.random.uniform(mutation_range[0], mutation_range[1])
            return np.clip(base_value + mutation, 0.1, 3.0)
        return base_value

    def apply_growth_gene(
        self, particle: CellularTypeData, gene_data: List[Any]
    ) -> None:
        """
        Handles juvenile-to-adult growth and sexual maturity.
        gene_data: [growth_rate, adult_size, maturity_age]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        gene_data : List[Any]
            Data for the growth gene.
        """
        # Default values if gene_data is incomplete
        growth_rate = gene_data[0] if len(gene_data) > 0 else 0.1
        adult_size = gene_data[1] if len(gene_data) > 1 else 2.0
        maturity_age = gene_data[2] if len(gene_data) > 2 else 100.0

        # Clamp values to reasonable ranges
        growth_rate = np.clip(growth_rate, 0.01, 0.5)
        adult_size = np.clip(adult_size, 1.0, 5.0)
        maturity_age = np.clip(maturity_age, 50.0, 200.0)

        # Calculate growth factor based on age
        juvenile_mask = particle.age < maturity_age
        growth_factor = np.where(
            juvenile_mask, growth_rate * (1.0 - particle.age / maturity_age), 0.0
        )

        # Apply growth effects
        particle.energy += growth_factor * particle.energy_efficiency

        if particle.mass_based and particle.mass is not None:
            # Grow mass for juvenile particles
            particle.mass = np.where(
                juvenile_mask, particle.mass * (1.0 + growth_factor), particle.mass
            )
            particle.mass = np.clip(particle.mass, 0.1, adult_size)

    def apply_predation_gene(
        self,
        particle: CellularTypeData,
        others: List[CellularTypeData],
        gene_data: List[Any],
        env: SimulationConfig,
    ) -> None:
        """
        Encodes predatory behavior for attacking and feeding on other particles.
        gene_data: [attack_power, energy_gain]

        Parameters:
        -----------
        particle : CellularTypeData
            The cellular type data of the particle.
        others : List[CellularTypeData]
            List of other cellular type data for interactions.
        gene_data : List[Any]
            Data for the predation gene.
        env : SimulationConfig
            Environmental configuration parameters.
        """
        # Default values if gene_data is incomplete
        attack_power = gene_data[0] if len(gene_data) > 0 else 10.0
        energy_gain = gene_data[1] if len(gene_data) > 1 else 5.0

        # Clamp values to reasonable ranges
        attack_power = np.clip(attack_power, 1.0, 20.0)
        energy_gain = np.clip(energy_gain, 1.0, 10.0)

        for other in others:
            if other == particle:
                continue

            # Calculate distances vectorized
            dx = other.x - particle.x[:, np.newaxis]
            dy = other.y - particle.y[:, np.newaxis]
            distances = np.sqrt(dx**2 + dy**2)

            # Create predation mask
            predation_mask = (
                (distances < env.predation_range)
                & other.alive[np.newaxis, :]
                & (particle.energy[:, np.newaxis] > other.energy)
            )

            if not np.any(predation_mask):
                continue

            # Get predator-prey pairs
            pred_idx, prey_idx = np.where(predation_mask)

            # Calculate damage based on attack power and relative energy levels
            energy_ratio = particle.energy[pred_idx] / other.energy[prey_idx]
            damage = attack_power * energy_ratio

            # Apply damage to prey
            other.energy[prey_idx] -= damage

            # Predators gain energy
            gained_energy = energy_gain * damage * particle.energy_efficiency[pred_idx]
            particle.energy[pred_idx] += gained_energy

            # Update prey alive status
            other.alive[prey_idx] = other.energy[prey_idx] > 0

            # Clamp energies to valid range
            particle.energy = np.clip(particle.energy, 0.0, 200.0)
            other.energy = np.clip(other.energy, 0.0, 200.0)
