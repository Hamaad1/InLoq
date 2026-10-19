#!/usr/bin/env python
"""
Standalone script to generate quantum circuit diagrams.
Run this independently to create circuit visualizations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits_visualization import generate_circuit_diagrams


def main():
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Set parameters
    N_QUBITS = 4  # Number of qubits (matches InLoQ default)
    LAYERS_TO_GENERATE = [1, 3]  # Layer depths you want
    
    # Output directory (following NeuroLoc structure)
    SAVE_DIR = PROJECT_ROOT / "paper" / "Plots" / "Circuit_Diagrams"
    
    print("="*70)
    print("QUANTUM CIRCUIT DIAGRAM GENERATOR")
    print("="*70)
    print(f"Qubits: {N_QUBITS}")
    print(f"Layers: {LAYERS_TO_GENERATE}")
    print(f"Output: {SAVE_DIR}")
    print("="*70)
    
    # Generate diagrams
    generate_circuit_diagrams(
        n_qubits=N_QUBITS,
        layers=LAYERS_TO_GENERATE,
        save_dir=str(SAVE_DIR)
    )
    
    print("\n" + "="*70)
    print("✅ Circuit diagram generation complete!")
    print(f"📁 Location: {SAVE_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()