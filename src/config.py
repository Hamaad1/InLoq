"""
src/config.py
==============
Configuration for InLoQ experiments.
Contains all hyperparameters, paths, and model selection flags.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results_InLoQ"

# Data file
CSV_PATH = DATA_DIR / "Inloq_redmi_fuldata.csv"

# ============================================================
# EXPERIMENT HYPERPARAMETERS
# ============================================================

# Training parameters
N_RUNS = 2          # Number of runs (10 for full experiment)
EPOCHS = 5          # Epochs per run (100 for full experiment)
BATCH_SIZE = 16

# ============================================================
# MODEL SELECTION FLAGS
# ============================================================

TRAIN_INLOQ = True
TRAIN_ANN = True
TRAIN_LSTM = True
TRAIN_GRU = True
TRAIN_BILSTM = True

# ============================================================
# SEED CONFIGURATION
# ============================================================

BASE_SEED = 42
MULTI_SEEDS = [42, 123, 456, 789, 101]  # For reproducibility

# ============================================================
# FEATURE SETS
# ============================================================

# FEATURE_SETS = {
#     # Define your feature sets here
#     # Example:
#     # 'all': ['rssi1', 'rssi2', 'rssi3', ...],
# }

# ============================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================

# ANN width matching
# The width will be auto-calculated to match InLoQ parameter count
# You can set a default here if needed
ANN_WIDTH_DEFAULT = 32

# LSTM/GRU/BiLSTM configurations
LSTM_UNITS = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2

GRU_UNITS = 64
GRU_LAYERS = 2
GRU_DROPOUT = 0.2

BILSTM_UNITS = 64
BILSTM_LAYERS = 2
BILSTM_DROPOUT = 0.2

# ============================================================
# RESULTS DIRECTORIES
# ============================================================

# This will be created automatically
INLOQ_RESULTS_DIR = RESULTS_DIR / "Inloq"
ANN_RESULTS_DIR = RESULTS_DIR / "Classical_ANN"
LSTM_RESULTS_DIR = RESULTS_DIR / "LSTM"
GRU_RESULTS_DIR = RESULTS_DIR / "GRU"
BILSTM_RESULTS_DIR = RESULTS_DIR / "BiLSTM"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_results_dir(model_type: str) -> Path:
    """
    Get the results directory for a specific model type.
    
    Args:
        model_type: 'Inloq', 'ANN', 'LSTM', 'GRU', 'BiLSTM'
    
    Returns:
        Path to the results directory
    """
    mapping = {
        'Inloq': INLOQ_RESULTS_DIR,
        'ANN': ANN_RESULTS_DIR,
        'LSTM': LSTM_RESULTS_DIR,
        'GRU': GRU_RESULTS_DIR,
        'BiLSTM': BILSTM_RESULTS_DIR,
    }
    return mapping.get(model_type, RESULTS_DIR / model_type)

def get_model_config(model_type: str) -> dict:
    """
    Get configuration for a specific model type.
    
    Args:
        model_type: 'Inloq', 'ANN', 'LSTM', 'GRU', 'BiLSTM'
    
    Returns:
        Dictionary with model-specific configuration
    """
    configs = {
        'Inloq': {
            'input_dim': None,  # Will be set at runtime
            'n_classes': 2,     # x and y coordinates
        },
        'ANN': {
            'width': ANN_WIDTH_DEFAULT,
            'activation': 'relu',
            'output_activation': 'linear',
        },
        'LSTM': {
            'units': LSTM_UNITS,
            'layers': LSTM_LAYERS,
            'dropout': LSTM_DROPOUT,
        },
        'GRU': {
            'units': GRU_UNITS,
            'layers': GRU_LAYERS,
            'dropout': GRU_DROPOUT,
        },
        'BiLSTM': {
            'units': BILSTM_UNITS,
            'layers': BILSTM_LAYERS,
            'dropout': BILSTM_DROPOUT,
        },
    }
    return configs.get(model_type, {})

def should_train_model(model_type: str) -> bool:
    """
    Check if a specific model should be trained.
    
    Args:
        model_type: 'Inloq', 'ANN', 'LSTM', 'GRU', 'BiLSTM'
    
    Returns:
        True if the model should be trained
    """
    mapping = {
        'Inloq': TRAIN_INLOQ,
        'ANN': TRAIN_ANN,
        'LSTM': TRAIN_LSTM,
        'GRU': TRAIN_GRU,
        'BiLSTM': TRAIN_BILSTM,
    }
    return mapping.get(model_type, False)

# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

def print_config():
    """Print the current configuration."""
    print("\n" + "=" * 70)
    print("InLoQ CONFIGURATION")
    print("=" * 70)
    print(f"Data path: {CSV_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    print("-" * 70)
    print(f"Runs: {N_RUNS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Base seed: {BASE_SEED}")
    print("-" * 70)
    print(f"Train InloQ: {'✅' if TRAIN_INLOQ else '❌'}")
    print(f"Train ANN: {'✅' if TRAIN_ANN else '❌'}")
    print(f"Train LSTM: {'✅' if TRAIN_LSTM else '❌'}")
    print(f"Train GRU: {'✅' if TRAIN_GRU else '❌'}")
    print(f"Train BiLSTM: {'✅' if TRAIN_BILSTM else '❌'}")
    print("=" * 70 + "\n")

# Auto-print config when imported
if __name__ != "__main__":
    pass  # Don't print on import, only when run directly