import random
from typing import Dict, List, Tuple, Union

import numpy as np
from gp_config import SimulationConfig

###############################################################
# Interaction Rules, Give-Take & Synergy
###############################################################


class InteractionRules:
    """
    Manages creation and evolution of interaction parameters between cellular types.

    This class handles the complex relationships between different cellular types,
    including force-based interactions, energy transfer mechanisms (give-take),
    and cooperative energy sharing (synergy). All relationships evolve over time
    to create dynamic, emergent behaviors in the simulation.

    Attributes:
        config: Configuration parameters for the simulation
        mass_based_type_indices: List of cellular type indices that use mass-based physics
        rules: Matrix of interaction parameters between each cellular type pair
        give_take_matrix: Boolean matrix defining energy transfer relationships
        synergy_matrix: Float matrix defining energy sharing relationship strengths
    """

    def __init__(
        self, config: SimulationConfig, mass_based_type_indices: List[int]
    ) -> None:
        """
        Initialize the InteractionRules with configuration and mass-based type indices.

        Args:
            config: Configuration parameters controlling simulation behavior
            mass_based_type_indices: Indices of cellular types that use mass-based physics
        """
        self.config = config
        self.mass_based_type_indices = mass_based_type_indices
        self.rules = self._create_interaction_matrix()
        self.give_take_matrix = self._create_give_take_matrix()
        self.synergy_matrix = self._create_synergy_matrix()

    def _create_interaction_matrix(
        self,
    ) -> List[Tuple[int, int, Dict[str, Union[bool, float]]]]:
        """
        Create a matrix of interaction parameters for all cellular type pairs.

        For each pair of cellular types (i,j), generates a parameter dictionary
        defining how they interact through forces, with specialized handling
        for mass-based types.

        Returns:
            List of tuples (i, j, params) where i and j are cellular type indices
            and params is a dictionary of interaction parameters
        """
        final_rules: List[Tuple[int, int, Dict[str, Union[bool, float]]]] = []

        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                params = self._random_interaction_params(i, j)
                final_rules.append((i, j, params))

        return final_rules

    def _random_interaction_params(
        self, i: int, j: int
    ) -> Dict[str, Union[bool, float]]:
        """
        Generate randomized interaction parameters between two cellular types.

        Creates a physics-based interaction model between cellular types i and j,
        with potential-based and/or gravity-based forces depending on their
        mass characteristics.

        Args:
            i: Index of the first cellular type
            j: Index of the second cellular type

        Returns:
            Dictionary of interaction parameters defining force calculations
        """
        # Determine if both types are mass-based for gravity interactions
        both_mass = (
            i in self.mass_based_type_indices and j in self.mass_based_type_indices
        )

        # 50% chance to use gravity if both types are mass-based
        use_gravity = both_mass and (random.random() < 0.5)

        # Always use potential-based interactions
        use_potential = True

        # Generate random potential strength within configured range
        potential_strength = random.uniform(
            self.config.interaction_strength_range[0],
            self.config.interaction_strength_range[1],
        )

        # 50% chance to invert the potential (attraction vs repulsion)
        if random.random() < 0.5:
            potential_strength = -potential_strength

        # Set gravity factor if applicable, otherwise zero
        gravity_factor = random.uniform(0.1, 2.0) if use_gravity else 0.0

        # Random interaction distance between 50-200 units
        max_dist = random.uniform(50.0, 200.0)

        # Compile interaction parameters
        params: Dict[str, Union[bool, float]] = {
            "use_potential": use_potential,
            "use_gravity": use_gravity,
            "potential_strength": potential_strength,
            "gravity_factor": gravity_factor,
            "max_dist": max_dist,
        }

        return params

    def _create_give_take_matrix(self) -> np.ndarray:
        """
        Create a matrix defining predator-prey energy transfer relationships.

        Generates a sparse boolean matrix where element (i,j) is True if
        cellular type i transfers energy to cellular type j during close
        encounters (prey-predator relationship).

        Returns:
            Boolean matrix of shape (n_cell_types, n_cell_types) defining
            give-take relationships between cellular types
        """
        # Initialize empty relationship matrix
        matrix = np.zeros(
            (self.config.n_cell_types, self.config.n_cell_types), dtype=bool
        )

        # For each non-identical cellular type pair
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j:
                    # 10% chance to establish give-take relationship
                    if random.random() < 0.1:
                        matrix[i, j] = True

        return matrix

    def _create_synergy_matrix(self) -> np.ndarray:
        """
        Create a matrix defining cooperative energy sharing relationships.

        Generates a sparse float matrix where element (i,j) represents the
        strength of energy sharing from type i to type j when they interact
        within synergy range. Higher values mean stronger sharing.

        Returns:
            Float matrix of shape (n_cell_types, n_cell_types) with values
            in range [0.0, 1.0] representing synergy strengths
        """
        # Initialize empty synergy matrix
        synergy_matrix = np.zeros(
            (self.config.n_cell_types, self.config.n_cell_types), dtype=float
        )

        # For each non-identical cellular type pair
        for i in range(self.config.n_cell_types):
            for j in range(self.config.n_cell_types):
                if i != j:
                    # 10% chance to establish synergy relationship
                    if random.random() < 0.1:
                        # Assign random synergy factor between 0.1 and 0.9
                        # Higher values create stronger energy sharing
                        synergy_matrix[i, j] = random.uniform(0.1, 0.9)
                    else:
                        # No synergy relationship
                        synergy_matrix[i, j] = 0.0

        return synergy_matrix

    def evolve_parameters(self, frame_count: int) -> None:
        """
        Evolve interaction parameters periodically to create dynamic relationships.

        Mutates interaction forces, energy transfer efficiencies, and synergy
        relationships at regular intervals defined by evolution_interval,
        creating evolutionary pressure and emergent behaviors.

        Args:
            frame_count: Current simulation frame number, used to determine
                when evolution should occur
        """
        # Only evolve at specified frame intervals
        if frame_count % self.config.evolution_interval != 0:
            return

        # Evolve interaction parameters with small random adjustments
        for _, _, params in self.rules:
            # 10% chance to adjust potential strength by ±5%
            if random.random() < 0.1:
                params["potential_strength"] *= random.uniform(0.95, 1.05)

            # 5% chance to adjust gravity factor if present
            if random.random() < 0.05 and "gravity_factor" in params:
                params["gravity_factor"] *= random.uniform(0.95, 1.05)

            # 5% chance to adjust interaction distance
            if random.random() < 0.05:
                # Ensure distance stays above minimum threshold
                params["max_dist"] = max(
                    10.0, params["max_dist"] * random.uniform(0.95, 1.05)
                )

        # Evolve global energy transfer efficiency
        if random.random() < 0.1:
            self.config.energy_transfer_factor = min(
                1.0,  # Maximum transfer efficiency cap
                self.config.energy_transfer_factor * random.uniform(0.95, 1.05),
            )

        # Evolve synergy relationships based on configured rate
        for i in range(self.synergy_matrix.shape[0]):
            for j in range(self.synergy_matrix.shape[1]):
                if random.random() < self.config.synergy_evolution_rate:
                    # Apply small random adjustment between -0.05 and +0.05
                    adjustment = random.random() * 0.1 - 0.05

                    # Adjust synergy factor, keeping it within valid range [0.0, 1.0]
                    self.synergy_matrix[i, j] = min(
                        1.0, max(0.0, self.synergy_matrix[i, j] + adjustment)
                    )
