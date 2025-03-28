from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from game_forge.src.gene_particles.gp_config import SimulationConfig

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


def random_xy(window_width: int, window_height: int, n: int = 1) -> np.ndarray:
    """Generate random position coordinates within window boundaries.

    Creates vectorized random positions for efficient particle initialization
    within the specified simulation window dimensions.

    Args:
        window_width: Width of simulation window in pixels
        window_height: Height of simulation window in pixels
        n: Number of coordinate pairs to generate

    Returns:
        np.ndarray: Array of shape (n, 2) containing random (x, y) coordinates

    Raises:
        AssertionError: If window dimensions are non-positive or n < 1
    """
    assert window_width > 0, "Window width must be positive"
    assert window_height > 0, "Window height must be positive"
    assert n > 0, "Number of points must be positive"

    return np.random.uniform(0, [window_width, window_height], (n, 2)).astype(float)


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

    colors: List[ColorRGB] = []
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
    params: Dict[str, Union[float, bool, np.ndarray]],
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
    dx = a_x - b_x
    dy = a_y - b_y
    d_sq = dx * dx + dy * dy

    # Early return: no force if components overlap or are beyond max_dist
    if d_sq == 0.0 or d_sq > params["max_dist"] ** 2:
        return 0.0, 0.0

    # Calculate distance and normalize direction vector
    d = math.sqrt(d_sq)
    dx_norm = dx / d
    dy_norm = dy / d

    # Initialize force components
    fx, fy = 0.0, 0.0

    # Apply potential-based interaction (repulsive)
    if params.get("use_potential", True):
        pot_strength = params.get("potential_strength", 1.0)
        F_pot = pot_strength / d
        fx += F_pot * dx_norm
        fy += F_pot * dy_norm

    # Apply gravity-based interaction (attractive)
    if params.get("use_gravity", False):
        if "m_a" in params and "m_b" in params:
            m_a = params["m_a"]
            m_b = params["m_b"]
            gravity_factor = params.get("gravity_factor", 1.0)
            F_grav = gravity_factor * (m_a * m_b) / d_sq
            fx -= F_grav * dx_norm  # Gravity pulls, not pushes
            fy -= F_grav * dy_norm

    return fx, fy


def give_take_interaction(
    giver_energy: np.ndarray,
    receiver_energy: np.ndarray,
    giver_mass: Optional[np.ndarray],
    receiver_mass: Optional[np.ndarray],
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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
            np.ndarray: Updated energy levels for receiving particles
            np.ndarray: Updated energy levels for giving particles
            Optional[np.ndarray]: New mass values for receiving particles (or None)
            Optional[np.ndarray]: New mass values for giving particles (or None)
    """
    # Calculate energy transfer amounts (scalar or array depending on input shapes)
    transfer_amount = receiver_energy * config.energy_transfer_factor

    # Execute energy transfer
    updated_receiver = receiver_energy - transfer_amount
    updated_giver = giver_energy + transfer_amount

    # Initialize mass variables with original values
    updated_receiver_mass = receiver_mass
    updated_giver_mass = giver_mass

    # Execute mass transfer if enabled and mass values are provided
    if config.mass_transfer and receiver_mass is not None and giver_mass is not None:
        mass_transfer_amount = receiver_mass * config.energy_transfer_factor
        updated_receiver_mass = receiver_mass - mass_transfer_amount
        updated_giver_mass = giver_mass + mass_transfer_amount

    return updated_giver, updated_receiver, updated_giver_mass, updated_receiver_mass


def apply_synergy(
    energyA: np.ndarray, energyB: np.ndarray, synergy_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
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
            np.ndarray: Updated energy levels for first component group
            np.ndarray: Updated energy levels for second component group
    """
    # Calculate average energy between component groups
    avg_energy = (energyA + energyB) * 0.5

    # Apply weighted average based on synergy factor
    newA = (energyA * (1.0 - synergy_factor)) + (avg_energy * synergy_factor)
    newB = (energyB * (1.0 - synergy_factor)) + (avg_energy * synergy_factor)

    return newA, newB
