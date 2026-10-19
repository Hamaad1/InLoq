"""
Quantum circuit visualization utilities.
Independent module for generating circuit diagrams.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pathlib import Path


def visualize_quantum_circuit(inputs, weights, n_qubits, n_layers, save_dir, 
                              title_prefix="InLoq_Circuit"):
    """
    Draws an attractive quantum circuit visualization using actual inputs and weights.
    
    Args:
        inputs: array of size (n_qubits,) - input values
        weights: array of size (n_layers, n_qubits, 3) - rotation angles
        n_qubits: number of qubits
        n_layers: number of quantum layers
        save_dir: directory to save the circuit diagram
        title_prefix: prefix for output filename
    """
    os.makedirs(save_dir, exist_ok=True)
    
    wire_labels = [f"$q_{{{i}}}$" for i in range(n_qubits)]
    dev = qml.device("default.qubit", wires=wire_labels)
    
    @qml.qnode(dev)
    def viz_circuit(inputs, weights):
        # Apply RY with actual input
        for i, w in enumerate(dev.wires):
            qml.RY(inputs[i] * np.pi, wires=w)
        
        # Apply Rot gates layer by layer
        for layer in range(n_layers):
            for i, w in enumerate(dev.wires):
                qml.Rot(*weights[layer, i], wires=w)
            
            # Apply CNOTs between adjacent qubits
            for i in range(n_qubits):
                if n_qubits > 1:
                    qml.CNOT(wires=[dev.wires[i], dev.wires[(i + 1) % n_qubits]])
                else:
                    qml.PhaseShift(np.pi/4, wires=dev.wires[i])
        
        return [qml.expval(qml.PauliZ(w)) for w in dev.wires]

    # Draw base circuit
    fig, ax = qml.draw_mpl(
        viz_circuit,
        fontsize=25,
        show_all_wires=True,
    )(inputs, weights)
    
    # Style the text
    for text in ax.texts:
        text_content = text.get_text()
        
        # Wire labels - make them prominent
        if text_content.startswith('$q_'):
            text.set_fontsize(25)
            text.set_fontweight('bold')
            text.set_color("#080808")
        
        # Gate labels - bold
        elif text_content in ['RY', 'Rot', 'CNOT']:
            text.set_fontsize(25)
            text.set_fontweight('bold')
            text.set_color("#050505")
        
        # Parameter values
        else:
            text.set_fontsize(25)
    
    # Adjust figure size
    fig.set_figwidth(15)
    fig.set_figheight(4 + n_qubits * 0.8)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    
    # Labels
    ax.set_xlabel("Quantum Operations", fontsize=18)
    ax.set_ylabel("Qubit Index", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()

    # Use this:
    try:
        plt.tight_layout()
    except:
        pass  # Skip tight_layout if it causes issues
    
    # Save the figure
    save_path = os.path.join(save_dir, f"{title_prefix}_{n_qubits}Q_{n_layers}L.pdf")
    plt.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=600)
    plt.close(fig)
    print(f"✅ Quantum circuit visualization saved: {save_path}")
    
    return save_path


def generate_circuit_diagrams(data_path=None, save_dir="paper/Plots/Circuit_Diagrams", 
                             n_qubits=4, layers=[1, 3]):
    """
    Generate circuit diagrams for specified layer depths.
    
    Args:
        data_path: Path to data file (optional, for realistic inputs)
        save_dir: Directory to save circuit diagrams
        n_qubits: Number of qubits to visualize
        layers: List of layer depths to generate
    """
    from pathlib import Path
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample inputs and weights
    np.random.seed(42)
    
    for n_layers in layers:
        print(f"\n🎨 Generating circuit diagram with {n_qubits} qubits, {n_layers} layers...")
        
        # Create sample input (normalized between 0 and 1)
        sample_input = np.random.rand(n_qubits).astype(np.float32)
        
        # Create sample weights (random rotation angles)
        weights = np.random.randn(n_layers, n_qubits, 3).astype(np.float32)
        
        # Generate and save circuit diagram
        output_path = visualize_quantum_circuit(
            inputs=sample_input,
            weights=weights,
            n_qubits=n_qubits,
            n_layers=n_layers,
            save_dir=str(save_path),
            title_prefix="InLoq_Circuit"
        )
        
        print(f"   ✅ Saved: {output_path}")
    
    print(f"\n✅ All circuit diagrams saved to: {save_dir}")


if __name__ == "__main__":
    # Standalone execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Generate circuit diagrams for 1 and 3 layers
    generate_circuit_diagrams(
        n_qubits=4,  # Typical number of qubits for InLoQ
        layers=[1, 3],
        save_dir="paper/Plots/Circuit_Diagrams"
    )