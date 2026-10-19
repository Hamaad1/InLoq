#!/usr/bin/env python
"""
Reproduce all figures from saved experiment results.
Loads data from results_InLoQ/ and generates plots in paper/Plots/
"""

import os
import sys
import json
import glob
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import FEATURE_SETS, set_global_seeds, BASE_SEED
from src.data_loader import load_and_clean_data
from src.circuits_visualization import generate_circuit_diagrams

# Import ONLY the functions that actually exist in src/plots.py
from src.plots import (
    # Main analysis functions (these are your entry points)
    run_multirun_analysis,
    run_all_models_comparison,
    run_multirun_comparison_1,
    plot_mean_comparison_overlay,
    analyze_predictions_multirun,
    save_aggregate_test_results,
    
    # Parameter analysis (separate from model training)
    plot_grouped_bar,
    plot_combined_InLoq_analysis,
)


def load_full_model_data(results_base_dir):
    """
    Load full model data in the format expected by plotting functions.
    This replicates the structure from run_experiments.py
    """
    all_model_data = {}
    
    # Define model directories and their keys
    model_configs = {
        'Inloq': os.path.join(results_base_dir, 'Inloq'),
        'Classical ANN': os.path.join(results_base_dir, 'Classical_ANN'),
        'LSTM': os.path.join(results_base_dir, 'LSTM'),
        'GRU': os.path.join(results_base_dir, 'GRU'),
        'BiLSTM': os.path.join(results_base_dir, 'BiLSTM'),
    }
    
    for model_name, model_dir in model_configs.items():
        if os.path.exists(model_dir):
            # Load aggregated results
            agg_path = os.path.join(model_dir, 'aggregated', '*_aggregated_results.csv')
            agg_files = glob.glob(agg_path)
            
            if agg_files:
                # Try to load the full pickled data if it exists
                pkl_path = os.path.join(model_dir, 'model_data.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        model_data = pickle.load(f)
                else:
                    # Create minimal structure from CSV
                    df = pd.read_csv(agg_files[0])
                    aggregated_results = {}
                    for _, row in df.iterrows():
                        feat_name = row['Feature_Set']
                        aggregated_results[feat_name] = {
                            'mae_mean': row['MAE_Mean'],
                            'mae_std': row['MAE_Std'],
                            'rmse_mean': row['RMSE_Mean'],
                            'rmse_std': row['RMSE_Std'],
                            'euclidean_mean': row['Euclidean_Mean'],
                            'euclidean_std': row['Euclidean_Std'],
                            'training_time_mean': row['Training_Time_Mean'],
                            'training_time_std': row['Training_Time_Std'],
                            'n_runs': row['N_Runs'],
                        }
                    
                    model_data = {
                        'aggregated_results': aggregated_results,
                        'all_runs_results': [],
                        'all_histories': [],
                        'all_errors': [],
                        'all_test_data': [],
                        'all_timings': [],
                        'sys_info': {},
                    }
                
                all_model_data[model_name] = model_data
                print(f"✅ Loaded {model_name}")
            else:
                print(f"⚠️ No aggregated results found for {model_name}")
    
    return all_model_data


def main():
    # Paths using Path for cross-platform compatibility
    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_BASE = PROJECT_ROOT / "results_InLoq"
    PAPER_PLOTS_DIR = PROJECT_ROOT / "paper" / "Plots"
    PAPER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Building image path
    BUILDING_IMG_PATH = PROJECT_ROOT / "data" / "building13.png"
    
    print("\n" + "="*70)
    print("Reproducing Figures from Saved Results")
    print("="*70)
    
    # 1. Generate quantum circuit diagrams
    print("\n" + "="*70)
    print("Generating quantum circuit diagrams...")
    print("="*70)
    
    generate_circuit_diagrams(
        n_qubits=4,
        layers=[1, 3],
        save_dir=str(PAPER_PLOTS_DIR / "Circuit_Diagrams")
    )
    
    # 2. Load all model data
    print("\n" + "="*70)
    print("Loading saved model results...")
    print("="*70)
    
    all_model_data = load_full_model_data(str(RESULTS_BASE))
    
    if not all_model_data:
        print("❌ No results found! Please run run_experiments.py first.")
        print("   Only circuit diagrams were generated.")
        return
    
    print(f"\n✅ Loaded {len(all_model_data)} models: {list(all_model_data.keys())}")
    
    # 3. Generate per-model analysis plots
    print("\n" + "="*70)
    print("Generating per-model analysis plots...")
    print("="*70)
    
    # Define results directories mapping
    results_dirs = {
        'Inloq': RESULTS_BASE / 'Inloq',
        'Classical ANN': RESULTS_BASE / 'Classical_ANN',
        'LSTM': RESULTS_BASE / 'LSTM',
        'GRU': RESULTS_BASE / 'GRU',
        'BiLSTM': RESULTS_BASE / 'BiLSTM',
    }
    
    for model_name, model_data in all_model_data.items():
        print(f"\n📊 Analyzing {model_name}...")
        
        # Check if model has necessary data
        if not model_data.get('all_histories') or not model_data.get('aggregated_results'):
            print(f"   ⚠️ Skipping {model_name} - insufficient data")
            continue
        
        res_dir = results_dirs.get(model_name, RESULTS_BASE / model_name.replace(' ', '_'))
        res_dir = str(res_dir)
        
        try:
            # This generates multiple plots per model
            run_multirun_analysis(model_data, res_dir, model_name)
            
            # Count generated PNG files
            if os.path.exists(res_dir):
                png_files = [f for f in os.listdir(res_dir) if f.endswith('.png')]
                print(f"   ✅ Generated {len(png_files)} PNG files in {res_dir}")
        except Exception as e:
            print(f"   ⚠️ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Generate comparison plots
    if len(all_model_data) >= 2:
        print("\n" + "="*70)
        print("Generating comparison plots...")
        print("="*70)
        
        comparison_dir = PAPER_PLOTS_DIR / 'Comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Get specific model data
        qnn_data = all_model_data.get('Inloq')
        ann_data = all_model_data.get('Classical ANN')
        
        # Get extra models (all except Inloq and ANN)
        extra_models = {k: v for k, v in all_model_data.items() 
                       if k not in ['Inloq', 'Classical ANN']}
        
        # 4a. Spatial analysis (per-reference point plots)
        print("\n🗺️ Generating spatial analysis plots...")
        try:
            if qnn_data and ann_data and BUILDING_IMG_PATH.exists():
                run_multirun_comparison_1(
                    qnn_data, ann_data, str(comparison_dir), str(BUILDING_IMG_PATH),
                    extra_models=extra_models if extra_models else None
                )
                print(f"   ✅ Spatial plots saved to {comparison_dir}")
                
                # Count feature set subdirectories
                for fs in FEATURE_SETS.keys():
                    fs_dir = comparison_dir / f"FeatureSet_{fs}"
                    if fs_dir.exists():
                        png_count = len(list(fs_dir.glob('*.png')))
                        print(f"   📍 {fs}: {png_count} spatial plots")
            else:
                print("   ⚠️ Skipping: need both Inloq and ANN data or building image")
        except Exception as e:
            print(f"   ⚠️ Error in spatial analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # 4b. Mean prediction overlay plots
        print("\n🔄 Generating mean prediction overlay plots...")
        try:
            if qnn_data and ann_data and BUILDING_IMG_PATH.exists():
                plot_mean_comparison_overlay(
                    qnn_data, ann_data, str(comparison_dir), str(BUILDING_IMG_PATH)
                )
                print(f"   ✅ Mean overlay plots saved to {comparison_dir}")
            else:
                print("   ⚠️ Skipping: need data")
        except Exception as e:
            print(f"   ⚠️ Error in mean overlay: {e}")
    
    # 5. Full model comparison (if more than 1 model)
    if len(all_model_data) > 1:
        print("\n" + "="*70)
        print("Generating full model comparison suite...")
        print("="*70)
        
        try:
            run_all_models_comparison(all_model_data, str(comparison_dir))
            all_models_dir = comparison_dir / 'all_models'
            if all_models_dir.exists():
                png_files = list(all_models_dir.glob('*.png'))
                print(f"   ✅ Generated {len(png_files)} PNG files in {all_models_dir}")
                csv_files = list(all_models_dir.glob('*.csv'))
                print(f"   ✅ Generated {len(csv_files)} CSV files in {all_models_dir}")
        except Exception as e:
            print(f"   ⚠️ Error in full comparison: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Generate parameter analysis plots (if needed)
    print("\n" + "="*70)
    print("Generating parameter analysis plots...")
    print("="*70)
    
    try:
        from src.parameter_counter import build_summary, INPUT_DIMS
        
        df_wide, df_long = build_summary(INPUT_DIMS)
        
        # Grouped bar chart
        plot_grouped_bar(df_long, str(PAPER_PLOTS_DIR / 'Parameter_Analysis'))
        
        # Combined InLoq analysis
        plot_combined_InLoq_analysis(INPUT_DIMS, str(PAPER_PLOTS_DIR / 'Parameter_Analysis'))
        
        print(f"   ✅ Parameter analysis plots saved to {PAPER_PLOTS_DIR / 'Parameter_Analysis'}")
    except Exception as e:
        print(f"   ⚠️ Could not generate parameter analysis: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL FIGURES GENERATED")
    print("="*70)
    print(f"📁 Location: {PAPER_PLOTS_DIR}")
    print("\nGenerated content:")
    print(f"  • Circuit diagrams: {PAPER_PLOTS_DIR / 'Circuit_Diagrams'}")
    print(f"  • Per-model plots: Individual model directories")
    print(f"  • Comparison plots: {PAPER_PLOTS_DIR / 'Comparison'}")
    print(f"  • Parameter analysis: {PAPER_PLOTS_DIR / 'Parameter_Analysis'}")
    print("="*70)


if __name__ == '__main__':
    main()