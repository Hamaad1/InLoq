"""
Utility functions: seeds, constants, system info
"""

import os
import random
import platform
import numpy as np
import tensorflow as tf
import pandas as pd

# Constants
FEATURE_SETS = {
    "BT only": ['BT'],
    "Bx, By, Bz": ['Bx', 'By', 'Bz'],
    "Bz, Bxy, BT": ['Bz', 'Bxy', 'BT'],
}
SEQUENCE_LENGTH = 6
BASE_SEED = 320


def set_global_seeds(seed: int) -> None:
    """
    Seed ALL random sources in one call.
    Call at program start, at the top of every run, and before every
    model build to guarantee full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_system_info(results_dir=None) -> dict:
    """Collect system information for reproducibility."""
    info = {
        'Python version': platform.python_version(),
        'Platform': platform.platform(),
        'Processor': platform.processor(),
        'TensorFlow version': tf.__version__,
    }
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            info.update({
                'GPU name': g.name,
                'Driver': g.driver,
                'Memory total (MB)': g.memoryTotal,
                'Memory used (MB)': g.memoryUsed,
                'Memory free (MB)': g.memoryFree,
                'GPU load (%)': round(g.load * 100, 2),
                'GPU temperature (C)': g.temperature,
            })
        else:
            info['GPU'] = 'No GPU detected'
    except Exception:
        tgpus = tf.config.list_physical_devices('GPU')
        info['GPU'] = str(tgpus) if tgpus else 'No GPU detected'

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        pd.DataFrame([info]).to_csv(
            os.path.join(results_dir, 'system_info.csv'), index=False)
        print(f'💾 System info -> {results_dir}/system_info.csv')

    return info


def count_params(model) -> int:
    """Count only trainable parameters correctly."""
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def get_model_info(model) -> dict:
    """Get detailed model information."""
    return {
        'total_params': model.count_params(),
        'trainable_params': count_params(model),
        'non_trainable_params': int(
            np.sum([np.prod(v.shape) for v in model.non_trainable_weights])),
        'layers': [
            {'name': l.name,
             'class': l.__class__.__name__,
             'output_shape': (l.output_shape if hasattr(l, 'output_shape') else None),
             'params': l.count_params()}
            for l in model.layers
        ]
    }

# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_full_summary(model_data: dict):
    print('\n' + '='*95)
    print(f'{"Model":<10} {"Feature Set":<22} {"MAE":>14} '
          f'{"RMSE":>14} {"Euclidean":>16} {"Train(s)":>10}')
    print('='*95)
    for model_name, data in model_data.items():
        for feat_name, agg in data['aggregated_results'].items():
            print(f'{model_name:<10} {feat_name:<22} '
                  f'{agg["mae_mean"]:>7.4f}+/-{agg["mae_std"]:.4f}  '
                  f'{agg["rmse_mean"]:>7.4f}+/-{agg["rmse_std"]:.4f}  '
                  f'{agg["euclidean_mean"]:>9.4f}+/-{agg["euclidean_std"]:.4f}  '
                  f'{agg["training_time_mean"]:>8.1f}')
    print('='*95)