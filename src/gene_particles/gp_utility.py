"""Gene Particles Utility Module.

Provides core utilities and data structures for the Gene Particles simulation system,
including trait definitions, vector operations, interaction physics, and energy transfer
mechanisms with mathematical precision and type safety.

This module implements:
    - Type-safe vector and physical calculations
    - Trait definition and validation framework
    - Energy transfer mechanics with conservation laws
    - Particle interaction physics with customizable force models
    - Color generation for visual distinction between particle types

All functions maintain dimensional consistency and type integrity across
vectorized operations for both performance and correctness guarantees.
"""

import math
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

import numpy as np

# Break circular import with TYPE_CHECKING conditional
if TYPE_CHECKING:
    from game_forge.src.gene_particles.gp_config import SimulationConfig
    from game_forge.src.gene_particles.gp_types import (
        BoolArray,
        ColorRGB,
        FloatArray,
        InteractionParams,
        Vector2D,
    )
else:
    # Direct type definitions to break import cycle
    Vector2D = Tuple[float, float]
    ColorRGB = Tuple[int, int, int]
    FloatArray = np.ndarray
    BoolArray = np.ndarray
    InteractionParams = dict


# Helper function for mutation calculations
def mutate_trait(
    base_values: "FloatArray",
    mutation_mask: "BoolArray",
    min_range: float,
    max_range: float,
) -> "FloatArray":
    """Generate mutated trait values from base values.

    Applies random mutations to trait values based on a mutation mask and range.

    Args:
        base_values: Original trait values to potentially mutate
        mutation_mask: Boolean mask indicating which values should be mutated
        min_range: Minimum mutation adjustment value
        max_range: Maximum mutation adjustment value

    Returns:
        FloatArray: New trait values with mutations applied where masked
    """
    values: "FloatArray" = base_values.copy()
    if np.any(mutation_mask):
        # Calculate number of mutations to apply based on mask
        num_mutations: int = int(np.sum(mutation_mask))

        # Generate mutation values within the specified range
        mutation: "FloatArray" = np.random.uniform(
            min_range, max_range, size=num_mutations
        ).astype(np.float64)

        # Apply mutations only to masked values
        values[mutation_mask] += mutation
    return values


def random_xy(window_width: int, window_height: int, n: int = 1) -> "FloatArray":
    """Generate random position coordinates within window boundaries.

    Creates vectorized random positions for efficient particle initialization
    within the specified simulation window dimensions.

    Args:
        window_width: Width of simulation window in pixels
        window_height: Height of simulation window in pixels
        n: Number of coordinate pairs to generate

    Returns:
        FloatArray: Array of shape (n, 2) containing random (x, y) coordinates

    Raises:
        AssertionError: If window dimensions are non-positive or n < 1
    """
    assert window_width > 0, "Window width must be positive"
    assert window_height > 0, "Window height must be positive"
    assert n > 0, "Number of points must be positive"

    # Generate uniform random coordinates and explicitly cast to ensure type safety
    coords: "FloatArray" = np.random.uniform(
        0, [window_width, window_height], (n, 2)
    ).astype(np.float64)

    return coords


def apply_growth_gene(
    energy: "FloatArray",
    growth_factor: float,
    min_energy: float,
    max_energy: float,
) -> "FloatArray":
    """Apply growth gene to energy levels.

    Modifies energy levels based on a growth factor, ensuring values remain
    within specified bounds.

    Args:
        energy: Current energy levels of particles
        growth_factor: Factor by which to increase energy
        min_energy: Minimum allowable energy level
        max_energy: Maximum allowable energy level

    Returns:
        FloatArray: Updated energy levels after applying growth factor
    """
    # Apply growth factor and clamp values to the specified range
    new_energy: "FloatArray" = np.clip(energy * growth_factor, min_energy, max_energy)
    return new_energy


def generate_vibrant_colors(n: int) -> List[ColorRGB]:
    """Generate distinct vibrant color tuples with maximum visual separation.

    Creates a palette of colors evenly distributed around the HSV color wheel,
    ensuring maximum distinguishability between different cellular types.

    Args:
        n: Number of distinct colors to generate

    Returns:
        List[ColorRGB]: List of RGB color tuples (r, g, b) with values 0-255

    Raises:
        AssertionError: If n is not positive
    """
    assert n > 0, "Number of colors must be positive"

    colors: List["ColorRGB"] = []
    for i in range(n):
        # Evenly distribute hues around the color wheel
        hue = (i / n) % 1.0

        # Calculate HSV to RGB conversion factors
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        q = int((1 - f) * 255)
        t = int(f * 255)
        v = 255  # Full value/brightness

        # Assign RGB based on the hue sector (one of six color wheel segments)
        if h_i == 0:
            r, g, b = v, t, 0
        elif h_i == 1:
            r, g, b = q, v, 0
        elif h_i == 2:
            r, g, b = 0, v, t
        elif h_i == 3:
            r, g, b = 0, q, v
        elif h_i == 4:
            r, g, b = t, 0, v
        else:  # h_i == 5
            r, g, b = v, 0, q

        colors.append((r, g, b))

    return colors


def apply_interaction(
    a_x: float,
    a_y: float,
    b_x: float,
    b_y: float,
    params: "InteractionParams",
) -> Vector2D:
    """Compute force exerted by cellular component B on A.

    Calculates potential and gravitational forces between two particles based
    on their positions and interaction parameters.

    Args:
        a_x: X-coordinate of cellular component A
        a_y: Y-coordinate of cellular component A
        b_x: X-coordinate of cellular component B
        b_y: Y-coordinate of cellular component B
        params: Dictionary containing interaction parameters:
            max_dist (float): Maximum interaction distance
            use_potential (bool): Whether to use potential-based forces
            potential_strength (float): Strength of potential interaction
            use_gravity (bool): Whether to use gravity-based forces
            m_a (float): Mass of component A (required for gravity)
            m_b (float): Mass of component B (required for gravity)
            gravity_factor (float): Gravity scaling factor

    Returns:
        Vector2D: Force vector (fx, fy) exerted on component A by component B
    """
    # Calculate displacement vector from B to A
    dx: float = a_x - b_x
    dy: float = a_y - b_y
    d_sq: float = dx * dx + dy * dy

    # Early return: no force if components overlap or are beyond max_dist
    max_dist = params["max_dist"]
    if d_sq == 0.0 or d_sq > max_dist**2:
        return 0.0, 0.0

    # Calculate distance and normalize direction vector
    d: float = math.sqrt(d_sq)
    dx_norm: float = dx / d
    dy_norm: float = dy / d

    # Initialize force components
    fx: float = 0.0
    fy: float = 0.0

    # Apply potential-based interaction (repulsive)
    if params.get("use_potential", True):
        pot_strength: float = params.get("potential_strength", 1.0)
        F_pot: float = pot_strength / d
        fx += F_pot * dx_norm
        fy += F_pot * dy_norm

    # Apply gravity-based interaction (attractive)
    if params.get("use_gravity", False):
        if "m_a" in params and "m_b" in params:
            m_a: float = cast(float, params["m_a"])
            m_b: float = cast(float, params["m_b"])
            gravity_factor: float = cast(float, params.get("gravity_factor", 1.0))
            F_grav: float = gravity_factor * (m_a * m_b) / d_sq
            fx -= F_grav * dx_norm  # Gravity pulls, not pushes
            fy -= F_grav * dy_norm

    return fx, fy


def give_take_interaction(
    giver_energy: "FloatArray",
    receiver_energy: "FloatArray",
    giver_mass: Optional["FloatArray"],
    receiver_mass: Optional["FloatArray"],
    config: "SimulationConfig",
) -> Tuple["FloatArray", "FloatArray", Optional["FloatArray"], Optional["FloatArray"]]:
    """Transfer energy and optionally mass from receiver to giver particles.

    Implements predator-prey energy transfer mechanism where "receiver" loses
    energy (and potentially mass) which is gained by the "giver". All operations
    maintain dimensional consistency across input arrays.

    Args:
        giver_energy: Energy levels of particles receiving energy transfer
        receiver_energy: Energy levels of particles giving up energy
        giver_mass: Mass values of receiving particles if mass-based; None otherwise
        receiver_mass: Mass values of giving particles if mass-based; None otherwise
        config: Configuration parameters controlling transfer amounts.
            Must have attributes:
                energy_transfer_factor (float): Portion of energy to transfer
                mass_transfer (bool): Whether to transfer mass along with energy

    Returns:
        Tuple containing:
            FloatArray: Updated energy levels for receiving particles
            FloatArray: Updated energy levels for giving particles
            Optional[FloatArray]: New mass values for receiving particles (or None)
            Optional[FloatArray]: New mass values for giving particles (or None)
    """
    # Calculate energy transfer amounts (scalar or array depending on input shapes)
    transfer_amount: "FloatArray" = receiver_energy * config.energy_transfer_factor

    # Execute energy transfer
    updated_receiver: "FloatArray" = receiver_energy - transfer_amount
    updated_giver: "FloatArray" = giver_energy + transfer_amount

    # Initialize mass variables with original values
    updated_receiver_mass: Optional["FloatArray"] = receiver_mass
    updated_giver_mass: Optional["FloatArray"] = giver_mass

    # Execute mass transfer if enabled and mass values are provided
    if config.mass_transfer and receiver_mass is not None and giver_mass is not None:
        mass_transfer_amount: "FloatArray" = (
            receiver_mass * config.energy_transfer_factor
        )
        updated_receiver_mass = receiver_mass - mass_transfer_amount
        updated_giver_mass = giver_mass + mass_transfer_amount

    return updated_giver, updated_receiver, updated_giver_mass, updated_receiver_mass


def apply_synergy(
    energyA: "FloatArray", energyB: "FloatArray", synergy_factor: float
) -> Tuple["FloatArray", "FloatArray"]:
    """Apply energy sharing between allied cellular components.

    Redistributes energy between components based on synergy factor,
    moving both components closer to their average energy level.

    Args:
        energyA: Energy levels of first component group
        energyB: Energy levels of second component group
        synergy_factor: Factor [0.0-1.0] determining energy sharing intensity
            0.0 = no sharing
            1.0 = complete equalization of energy

    Returns:
        Tuple containing:
            FloatArray: Updated energy levels for first component group
            FloatArray: Updated energy levels for second component group
    """
    # Calculate average energy between component groups
    avg_energy: "FloatArray" = (energyA + energyB) * 0.5

    # Apply weighted average based on synergy factor
    newA: "FloatArray" = (energyA * (1.0 - synergy_factor)) + (
        avg_energy * synergy_factor
    )
    newB: "FloatArray" = (energyB * (1.0 - synergy_factor)) + (
        avg_energy * synergy_factor
    )

    return newA, newB


if __name__ == "__main__":
    """Module demonstration when run directly."""
    print("🧬 Gene Particles Utility Module")
    print("This module provides utility functions for the Gene Particles simulation.")
    print("It is not intended to be run directly, but to be imported by other modules.")

    # Simple demonstration of color generation
    print("\nDemonstrating vibrant color generation:")
    demo_colors = generate_vibrant_colors(5)
    for i, color in enumerate(demo_colors):
        print(f"  Color {i+1}: RGB{color}")

    # Demonstrate vector operations
    print("\nDemonstrating random position generation:")
    positions = random_xy(100, 100, 3)
    for i, pos in enumerate(positions):
        print(f"  Position {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")

    print("\nTo use this module, import it into your Gene Particles simulation.")
