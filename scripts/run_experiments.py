#!/usr/bin/env python
"""
Main entry point for training models and saving results.
This script ONLY trains models and saves results to results_InLoQ/.
No plotting or table generation happens here.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so src can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_global_seeds, BASE_SEED, FEATURE_SETS
from src.data_loader import load_and_clean_data
from src.models import build_InLoq
from src.training import run_multiple_training_iterations, save_aggregated_results
from src.evaluation import compute_aggregated_results


def find_matching_ann_width(input_dim, target_params):
    """Find ANN width that matches target parameter count."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    for w in range(8, 512, 2):
        test_model = Sequential([
            Dense(w, input_dim=input_dim, activation='relu'),
            Dense(w // 2, activation='relu'),
            Dense(2, activation='linear')
        ])
        if test_model.count_params() >= target_params:
            print(f"Matched ANN Width found: {w}")
            print(f"-> ANN Params: {test_model.count_params()} vs Inloq Params: {target_params}")
            return w
    return 32  # default


def main():
    # Get the project root directory (where this script is located)
    PROJECT_ROOT = Path(__file__).parent.parent
    CSV_PATH = PROJECT_ROOT / "data" / "Inloq_redmi_fuldata.csv"
    BASE_DIR = PROJECT_ROOT / "results_InLoq"  # ← Define once here
    
    # Check if data file exists
    if not CSV_PATH.exists():
        print(f"❌ Error: Data file not found at {CSV_PATH}")
        print("Please ensure the data file exists in the 'data/' directory")
        return
    
    # Experiment hyperparameters
    N_RUNS = 2  # 10 for full experiment
    EPOCHS = 5  # 100 for full experiment
    BATCH_SIZE = 16
    
    # Model selection flags
    TRAIN_INLOQ = True
    TRAIN_ANN = True
    TRAIN_LSTM = True
    TRAIN_GRU = True
    TRAIN_BILSTM = True
    
    # Set master seed
    set_global_seeds(BASE_SEED)
    
    # Create directories
    results_dirs = {}
    if TRAIN_INLOQ:
        results_dirs['Inloq'] = BASE_DIR / 'Inloq'
        results_dirs['Inloq'].mkdir(parents=True, exist_ok=True)
    if TRAIN_ANN:
        results_dirs['ANN'] = BASE_DIR / 'Classical_ANN'
        results_dirs['ANN'].mkdir(parents=True, exist_ok=True)
    if TRAIN_LSTM:
        results_dirs['LSTM'] = BASE_DIR / 'LSTM'
        results_dirs['LSTM'].mkdir(parents=True, exist_ok=True)
    if TRAIN_GRU:
        results_dirs['GRU'] = BASE_DIR / 'GRU'
        results_dirs['GRU'].mkdir(parents=True, exist_ok=True)
    if TRAIN_BILSTM:
        results_dirs['BiLSTM'] = BASE_DIR / 'BiLSTM'
        results_dirs['BiLSTM'].mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {CSV_PATH}...")
    cleaned_df = load_and_clean_data(str(CSV_PATH))  # Convert Path to string for pandas
    
    # Determine input dimension and match ANN width
    target_cols = ['x', 'y']
    feature_cols = [c for c in cleaned_df.columns if c not in target_cols]
    input_dim = len(feature_cols)
    print(f"\nInput Dimension: {input_dim}")
    
    print("Calculating Inloq complexity for fair comparison...")
    InLoq_dummy = build_InLoq(input_dim=input_dim)
    target_params = InLoq_dummy.count_params()
    print(f"Target Trainable Parameters (Inloq): {target_params}")
    
    matched_width = find_matching_ann_width(input_dim, target_params)
    
    # Override ANN builder to use matched width
    if TRAIN_ANN:
        from src.models import build_ann
        # Monkey patch or create a wrapper - we'll handle this in the training function
    
    # Dictionary to store all model data (will be saved, not plotted)
    all_model_data = {}
    
    # Train Inloq
    if TRAIN_INLOQ:
        print('\n' + '='*70)
        print('Training Inloq (Quantum Neural Network)')
        print('='*70)
        qnn_data = run_multiple_training_iterations(
            cleaned_df, model_type='Inloq',
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            n_runs=N_RUNS, verbose=1,
            results_dir=str(results_dirs['Inloq']))  # Convert Path to string
        save_aggregated_results(qnn_data, str(results_dirs['Inloq']), 'Inloq')
        
        # Save a metadata file
        import json
        metadata = {
            'model_type': 'Inloq',
            'n_runs': N_RUNS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'base_seed': BASE_SEED,
            'input_dim': input_dim,
            'feature_sets': list(FEATURE_SETS.keys()),
        }
        with open(results_dirs['Inloq'] / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Train Classical ANN
    if TRAIN_ANN:
        print('\n' + '='*70)
        print(f'Training Classical ANN (Matched Width: {matched_width})')
        print('='*70)
        # Use a custom builder with matched width
        from src.models import build_ann
        import src.models as models_module
        
        # Temporarily replace build_ann with width-matched version
        original_build_ann = models_module.build_ann
        def build_ann_matched(input_dim, seed):
            return build_ann(input_dim, seed, width=matched_width)
        models_module.build_ann = build_ann_matched
        
        ann_data = run_multiple_training_iterations(
            cleaned_df, model_type='ANN',
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            n_runs=N_RUNS, verbose=1,
            results_dir=str(results_dirs['ANN']))
        save_aggregated_results(ann_data, str(results_dirs['ANN']), 'Classical')
        
        # Restore original
        models_module.build_ann = original_build_ann
    
    # Train LSTM
    if TRAIN_LSTM:
        print('\n' + '='*70)
        print('Training LSTM')
        print('='*70)
        lstm_data = run_multiple_training_iterations(
            cleaned_df, model_type='LSTM',
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            n_runs=N_RUNS, verbose=1,
            results_dir=str(results_dirs['LSTM']))
        save_aggregated_results(lstm_data, str(results_dirs['LSTM']), 'LSTM')
    
    # Train GRU
    if TRAIN_GRU:
        print('\n' + '='*70)
        print('Training GRU')
        print('='*70)
        gru_data = run_multiple_training_iterations(
            cleaned_df, model_type='GRU',
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            n_runs=N_RUNS, verbose=1,
            results_dir=str(results_dirs['GRU']))
        save_aggregated_results(gru_data, str(results_dirs['GRU']), 'GRU')
    
    # Train BiLSTM
    if TRAIN_BILSTM:
        print('\n' + '='*70)
        print('Training BiLSTM')
        print('='*70)
        bilstm_data = run_multiple_training_iterations(
            cleaned_df, model_type='BiLSTM',
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            n_runs=N_RUNS, verbose=1,
            results_dir=str(results_dirs['BiLSTM']))
        save_aggregated_results(bilstm_data, str(results_dirs['BiLSTM']), 'BiLSTM')
    
    print('\n' + '='*70)
    print('Training completed! Results saved to:')
    for model_name, dir_path in results_dirs.items():
        print(f'  {model_name}: {dir_path}')
    print('='*70)
    print('\nTo generate figures and tables, run:')
    print('  python scripts/reproduce_figures.py')
    print('  python scripts/reproduce_tables.py')
    # print('  python scripts/generate_circuit_diagrams.py')
    print('  python scripts/analyze_parameters.py')
    print('  python scripts/analyze_gradient_variance.py')


if __name__ == '__main__':
    main()