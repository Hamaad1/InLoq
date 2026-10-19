#!/usr/bin/env python
"""
Ablation Study: Quantum Circuit Depth Effect on Gradient Variance
(Barren Plateau Analysis)

This script analyzes how circuit depth affects gradient variance in quantum layers.
Independent of the main training pipeline.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing modules
from src.utils import set_global_seeds, BASE_SEED
from src.data_loader import load_and_clean_data, prepare_data
from src.models import _ki, QuantumLayer
from src.parameter_counter import INPUT_DIMS  # Use existing constants

# Import variable-depth quantum layer from ablation script
from src.quantum_varied import QuantumLayerVarDepth  # We'll create this


def compute_gradient_variance(model, X_sample, y_sample) -> float:
    """
    Compute mean variance of gradients across all trainable weights
    for a single forward + backward pass.
    """
    X_t = tf.cast(X_sample, tf.float32)
    y_t = tf.cast(y_sample, tf.float32)
    
    with tf.GradientTape() as tape:
        preds = model(X_t, training=True)
        loss = tf.reduce_mean(tf.square(preds - y_t))
    
    grads = tape.gradient(loss, model.trainable_variables)
    vars_ = [tf.math.reduce_variance(g).numpy() 
             for g in grads if g is not None]
    
    return float(np.mean(vars_)) if vars_ else 0.0


def analyze_gradient_variance(
    cleaned_df,
    results_dir: str,
    feature_set_name: str = "BT only",
    n_repeats: int = 5,
    depths: list = [1, 2, 4, 6],
    n_qubits: int = 4
):
    """
    Analyze gradient variance for different quantum circuit depths.
    
    Args:
        cleaned_df: Loaded and cleaned dataframe
        results_dir: Directory to save results
        feature_set_name: Which feature set to use (default: "BT only" for speed)
        n_repeats: Number of repetitions per depth
        depths: List of circuit depths to test
        n_qubits: Number of qubits to use (fixed for this analysis)
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Get features for the chosen set
    from src.utils import FEATURE_SETS
    features = FEATURE_SETS[feature_set_name]
    
    print("\n" + "="*70)
    print("QUANTUM CIRCUIT DEPTH: GRADIENT VARIANCE ANALYSIS")
    print("="*70)
    print(f"Feature set: {feature_set_name} (features: {features})")
    print(f"n_qubits: {n_qubits}")
    print(f"Depths to test: {depths}")
    print(f"Repeats per depth: {n_repeats}")
    
    # Prepare fixed data sample
    set_global_seeds(BASE_SEED)
    X_tr, _, _, y_tr, _, _, _ = prepare_data(
        features, cleaned_df, 
        cleaned_df[['x_coord', 'y_coord']], 
        random_state=BASE_SEED
    )
    
    X_sample = X_tr[:32].astype(np.float32)
    y_sample = y_tr[:32].astype(np.float32)
    
    rows = []
    
    for depth in depths:
        print(f"\n  Testing depth = {depth}...")
        grad_variances = []
        training_times = []
        
        for rep in range(n_repeats):
            set_global_seeds(BASE_SEED + rep)
            
            # Build model with variable depth
            model = Sequential([
                Dense(n_qubits, input_dim=len(features), activation='tanh',
                      kernel_initializer=_ki(BASE_SEED + rep)),
                QuantumLayerVarDepth(
                    n_qubits=n_qubits, 
                    input_dim=n_qubits,
                    n_layers=depth,
                    seed=BASE_SEED + rep,
                    name='quantum_layer'
                ),
                Dense(2, activation='linear',
                      kernel_initializer=_ki(BASE_SEED + rep + 1)),
            ])
            model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
            
            # Measure gradient variance
            t0 = time.time()
            gv = compute_gradient_variance(model, X_sample, y_sample)
            elapsed = time.time() - t0
            
            grad_variances.append(gv)
            training_times.append(elapsed)
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
        
        gv_mean = float(np.mean(grad_variances))
        gv_std = float(np.std(grad_variances))
        plateau_status = '⚠️ Possible barren plateau' if gv_mean < 1e-6 else '✓ Trainable'
        
        print(f"    depth={depth}: grad_var={gv_mean:.3e} ± {gv_std:.3e}  {plateau_status}")
        
        rows.append({
            'Depth': depth,
            'n_qubits': n_qubits,
            'Quantum_Params': depth * n_qubits * 3,
            'GradVar_Mean': gv_mean,
            'GradVar_Std': gv_std,
            'Status': plateau_status,
            'Mean_Time_sec': np.mean(training_times),
            'Note': '← main model' if depth == 2 else '',
        })
    
    # Save results
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = os.path.join(results_dir, 'ablation5_gradient_variance.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Generate plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Gradient variance vs depth
        ax1.errorbar(df['Depth'], df['GradVar_Mean'], yerr=df['GradVar_Std'],
                    marker='o', capsize=5, capthick=2, markersize=8,
                    linestyle='-', linewidth=2, color='purple')
        ax1.set_xlabel('Circuit Depth (layers)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Gradient Variance', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('Barren Plateau Analysis', fontsize=12, fontweight='bold')
        
        # Add threshold line
        ax1.axhline(y=1e-6, color='red', linestyle='--', alpha=0.7, 
                   label='Barren plateau threshold')
        ax1.legend()
        
        # Plot 2: Quantum parameters vs depth
        ax2.bar(df['Depth'], df['Quantum_Params'], color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Circuit Depth (layers)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Quantum Parameters', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_title('Parameter Count Scaling', fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for i, (depth, params) in enumerate(zip(df['Depth'], df['Quantum_Params'])):
            ax2.text(depth, params + 5, str(params), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        plot_path = os.path.join(results_dir, 'gradient_variance_analysis.pdf')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Plot saved to: {plot_path}")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plot generation")
    
    return df


def main():
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "Inloq_redmi_fuldata.csv"
    
    # Create output directory
    results_dir = PROJECT_ROOT / "paper" / "Plots" / "Ablation_Studies"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {CSV_PATH}...")
    cleaned_df = load_and_clean_data(str(CSV_PATH))
    
    # Run gradient variance analysis
    df_results = analyze_gradient_variance(
        cleaned_df=cleaned_df,
        results_dir=str(results_dir),
        feature_set_name="BT only",  # Fastest for analysis
        n_repeats=5,
        depths=[1, 2, 4, 6, 8],  # Test more depths
        n_qubits=4
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Gradient Variance Analysis")
    print("="*70)
    print(df_results.to_string())
    
    print("\n" + "="*70)
    print(f"✅ Analysis complete! Results saved to: {results_dir}")
    print("="*70)


if __name__ == '__main__':
    main()