"""
Evaluation metrics and aggregation
"""

import numpy as np


def compute_aggregated_results(all_runs_results, all_timings, feature_sets) -> dict:
    """
    Compute aggregated results across all runs.
    """
    agg = {}
    for feat_name in feature_sets:
        loss_l, mae_l, rmse_l, euc_l, std_euc_l = [], [], [], [], []
        train_l, pred_l = [], []
        params_l = []

        for run_res, run_tim in zip(all_runs_results, all_timings):
            if feat_name in run_res and isinstance(run_res[feat_name], dict):
                r = run_res[feat_name]
                loss_l.append(r['loss'])
                mae_l.append(r['mae'])
                rmse_l.append(r['rmse'])
                euc_l.append(r['mean_euclidean'])
                std_euc_l.append(r['std_euclidean'])
                if 'n_params' in r:
                    params_l.append(r['n_params'])
            if feat_name in run_tim:
                train_l.append(run_tim[feat_name]['training_time_sec'])
                pred_l.append(run_tim[feat_name]['prediction_time_sec'])
        
        agg[feat_name] = {
            'loss_mean': float(np.mean(loss_l)) if loss_l else 0.0,
            'loss_std': float(np.std(loss_l)) if loss_l else 0.0,
            'mae_mean': float(np.mean(mae_l)) if mae_l else 0.0,
            'mae_std': float(np.std(mae_l)) if mae_l else 0.0,
            'rmse_mean': float(np.mean(rmse_l)) if rmse_l else 0.0,
            'rmse_std': float(np.std(rmse_l)) if rmse_l else 0.0,
            'euclidean_mean': float(np.mean(euc_l)) if euc_l else 0.0,
            'euclidean_std': float(np.std(euc_l)) if euc_l else 0.0,
            'euclidean_std_mean': float(np.mean(std_euc_l)) if std_euc_l else 0.0,
            'training_time_mean': float(np.mean(train_l)) if train_l else 0.0,
            'training_time_std': float(np.std(train_l)) if train_l else 0.0,
            'prediction_time_mean': float(np.mean(pred_l)) if pred_l else 0.0,
            'prediction_time_std': float(np.std(pred_l)) if pred_l else 0.0,
            'n_params_mean': float(np.mean(params_l)) if params_l else 0.0,
            'n_runs': len(mae_l),
        }

    return agg


def compute_metrics(y_true, y_pred):
    """Compute all metrics for predictions."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    euc_distances = np.linalg.norm(y_pred - y_true, axis=1)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mean_euclidean': np.mean(euc_distances),
        'std_euclidean': np.std(euc_distances),
    }