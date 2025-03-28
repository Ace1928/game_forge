"""
GeneParticles: Advanced Cellular Automata with Dynamic Gene Expression, Emergent Behaviors,
and Extended Complexity
-------------------------------------------------------------------------------------------------
A hyper-advanced particle simulation that models cellular-like entities ("particles") endowed with
complex dynamic genetic traits, adaptive behaviors, emergent properties, hierarchical speciation,
and intricate interaction networks spanning multiple dimensions of trait synergy and competition.

This simulation implements:
1. Dynamic Gene Expression & Hyper-Complex Heredity
    - Multiple mutable traits influencing behavior and reproduction
    - Hierarchical gene clusters with layered mutation strategies
    - Nonlinear genotype-to-phenotype mappings with environmental feedback

2. Adaptive Population Management & Homeostasis
    - Real-time FPS monitoring with optimization triggers
    - Multi-factor fitness-based culling
    - Dynamic population growth based on resource availability

3. Enhanced Evolutionary Mechanisms & Speciation
    - Resource competition with synergy-influenced survival
    - Multidimensional genetic drift triggering speciation events
    - Complete lineage tracking with phylogenetic trees

4. Multi-Scale Complex Interactions
    - Force-based dynamics (potential, gravitational, synergistic)
    - Conditional energy routing based on species relationships
    - Emergent flocking, predation, and colony behaviors

5. Vectorized Performance Optimization
    - NumPy-based computation for all operations
    - Multi-level spatial partitioning using KD-trees
    - Adaptive rendering and parameterized update frequencies

Technical Requirements:
---------------------
- Python 3.8+
- NumPy >= 1.20.0
- Pygame >= 2.0.0
- SciPy >= 1.7.0

Installation:
------------
pip install numpy pygame scipy

Usage:
------
python geneparticles.py

Controls:
- ESC: Exit simulation
"""

from gp_automata import CellularAutomata
from gp_config import SimulationConfig

###############################################################
# Entry Point
###############################################################


def main():
    """Initialize and run the Gene Particles simulation.

    Creates a simulation with default configuration parameters and executes
    the main evolutionary loop until termination conditions are met.

    The simulation continues until one of the following occurs:
        - User presses the ESC key
        - User closes the simulation window
        - Simulation reaches configured maximum frame count

    Returns:
        NoReturn: Function does not return normally, terminates via system exit

    Raises:
        ImportError: If required dependencies are not installed
        RuntimeError: If simulation initialization fails
    """
    # Create simulation configuration with default parameters
    config: SimulationConfig = SimulationConfig()

    # Initialize the cellular automata system with configuration
    cellular_automata: CellularAutomata = CellularAutomata(config)

    # Execute main simulation loop until termination
    cellular_automata.main_loop()


if __name__ == "__main__":
    main()
