"""
Training loops with multi-run support
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.utils import set_global_seeds, get_system_info, count_params, get_model_info, FEATURE_SETS, SEQUENCE_LENGTH, BASE_SEED
from src.data_loader import prepare_data, prepare_sequences
from src.models import build_model
# , get_model_builder
from src.evaluation import compute_aggregated_results
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _train_one_model_all_features(
        cleaned_df: pd.DataFrame,
        model_name: str,
        epochs: int,
        batch_size: int,
        seed: int,
        verbose: int = 0,
        results_dir: str = None):
    """
    Train <model_name> on every feature set for one seed.
    """
    set_global_seeds(seed)
    sys_info = get_system_info(results_dir=results_dir)

    if verbose >= 1:
        print(f'  [{model_name}] seed={seed}')

    y_target = cleaned_df[['x_coord', 'y_coord']]

    histories_dict = {}
    results_dict   = {}
    errors_dict    = {}
    test_data_dict = {}
    timings_dict   = {}

    try:
        for feat_name, features in FEATURE_SETS.items():
            set_global_seeds(seed)          # re-seed before every feature block
            input_dim = len(features)

            try:
                X_tr, X_va, X_te, y_tr, y_va, y_te, _ = prepare_data(
                    features, cleaned_df, y_target, random_state=seed)

                is_recurrent = model_name in ('LSTM', 'GRU', 'BiLSTM')
                if is_recurrent:
                    Xtr, ytr = prepare_sequences(X_tr, y_tr, SEQUENCE_LENGTH)
                    Xva, yva = prepare_sequences(X_va, y_va, SEQUENCE_LENGTH)
                    Xte, yte = prepare_sequences(X_te, y_te, SEQUENCE_LENGTH)
                else:
                    Xtr, ytr = X_tr, y_tr
                    Xva, yva = X_va, y_va
                    Xte, yte = X_te, y_te

                model    = build_model(model_name, input_dim, seed)
                n_params = count_params(model)

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=100,
                                  restore_best_weights=True, verbose=1)
                ]

                t0 = time.time()
                history = model.fit(
                    Xtr, ytr,
                    validation_data=(Xva, yva),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True, 
                    callbacks=callbacks,
                    verbose=1
                )
                train_time = time.time() - t0

                t1        = time.time()
                y_pred    = model.predict(Xte, verbose=1)
                pred_time = time.time() - t1

                # ── metrics ──────────────────────────────────────────────────────
                test_loss, test_mae = model.evaluate(Xte, yte, verbose=1)
                euc          = np.linalg.norm(y_pred - yte, axis=1)
                from sklearn.metrics import mean_squared_error
                rmse_overall = float(np.sqrt(mean_squared_error(yte, y_pred)))
                # rmse_overall = float(np.sqrt(mean_squared_error(yte, y_pred)))
                mean_euc     = float(np.mean(euc))
                std_euc      = float(np.std(euc))
                model_info_d = get_model_info(model)

                if verbose >= 1:
                    print(f'    [{feat_name}] '
                          f'MAE={test_mae:.4f}  RMSE={rmse_overall:.4f}  '
                          f'Euc={mean_euc:.4f}  params={n_params}  t={train_time:.1f}s')

                # ── store ───────────────────────────────────────────────────────
                histories_dict[feat_name] = history
                results_dict[feat_name] = {
                    'Model'          : feat_name,
                    'loss'           : float(test_loss),
                    'mae'            : float(test_mae),
                    'rmse'           : rmse_overall,
                    'mean_euclidean' : mean_euc,
                    'std_euclidean'  : std_euc,
                    'model_info'     : model_info_d,
                    'n_params'       : n_params,
                }
                errors_dict[feat_name]    = euc
                test_data_dict[feat_name] = {'y_test': yte, 'y_pred': y_pred}
                timings_dict[feat_name] = {
                    'training_time_sec'  : train_time,
                    'prediction_time_sec': pred_time,
                }

                del model
                tf.keras.backend.clear_session()

            except KeyboardInterrupt:
                print(f'\n⚠️ KeyboardInterrupt detected while training {model_name} on {feat_name}')
                print('   Cleaning up and exiting...')
                raise  # Re-raise to stop execution
    
    except KeyboardInterrupt:
        print(f'\n❌ Training interrupted by user for {model_name}')
        # Clean up TensorFlow session
        tf.keras.backend.clear_session()
        raise  # Re-raise to stop the main script

    return histories_dict, results_dict, errors_dict, test_data_dict, timings_dict, sys_info


def run_multiple_training_iterations(
        cleaned_df: pd.DataFrame,
        model_type: str,
        epochs: int,
        batch_size: int,
        n_runs: int,
        verbose: int,
        results_dir: str) -> dict:
    """
    Run <model_type> for n_runs (seeds BASE_SEED ... BASE_SEED+n_runs-1).
    """
    os.makedirs(results_dir, exist_ok=True)

    all_histories = []
    all_runs_results = []
    all_errors = []
    all_test_data = []
    all_timings = []
    sys_info = {}

    print(f'\n{"="*70}')
    print(f'Runs {n_runs} (seeds {BASE_SEED}-{BASE_SEED+n_runs-1}) model={model_type}')
    print(f'{"="*70}')

    for run_idx in range(n_runs):
        seed = BASE_SEED + run_idx
        print(f'\n{"="*70}')
        print(f'Run {run_idx+1}/{n_runs}   seed={seed}')
        print(f'{"="*70}')

        set_global_seeds(seed)
        hist, res, err, td, tim, si = _train_one_model_all_features(
            cleaned_df, model_type, epochs, batch_size,
            seed=seed, verbose=verbose,
            results_dir=(results_dir if run_idx == 0 else None))

        if not sys_info:
            sys_info = si

        all_histories.append(hist)
        all_runs_results.append(res)
        all_errors.append(err)
        all_test_data.append(td)
        all_timings.append(tim)

        # Save run CSV immediately
        _save_run_csv(res, tim, run_idx + 1, seed, results_dir, model_type)

        print(f'\nRun {run_idx+1} Summary:')
        for fn, r in res.items():
            if isinstance(r, dict):
                print(f'  {fn}: MAE={r["mae"]:.4f} RMSE={r["rmse"]:.4f} '
                      f'Euclidean={r["mean_euclidean"]:.4f}')

    aggregated_results = compute_aggregated_results(all_runs_results, all_timings, FEATURE_SETS)

    print(f'\nAGGREGATED RESULTS ({model_type})')
    print('='*70)
    for fn, agg in aggregated_results.items():
        print(f'\n{fn} (n={agg["n_runs"]} runs)')
        print(f'  MAE: {agg["mae_mean"]:.4f} +/- {agg["mae_std"]:.4f}')
        print(f'  RMSE: {agg["rmse_mean"]:.4f} +/- {agg["rmse_std"]:.4f}')
        print(f'  Euclidean: {agg["euclidean_mean"]:.4f} +/- {agg["euclidean_std"]:.4f}')
        print(f'  Train(s): {agg["training_time_mean"]:.2f} +/- {agg["training_time_std"]:.2f}')

    return {
        'aggregated_results': aggregated_results,
        'all_runs_results': all_runs_results,
        'all_histories': all_histories,
        'all_errors': all_errors,
        'all_test_data': all_test_data,
        'all_timings': all_timings,
        'sys_info': sys_info,
    }


def _save_run_csv(res, tim, run_num: int, seed: int, results_dir: str, model_type: str):
    """Save one run's metrics immediately after training."""
    import pandas as pd
    runs_dir = os.path.join(results_dir, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    rows = []
    for feat_name, m in res.items():
        if not isinstance(m, dict):
            continue
        row = {
            'Run': run_num,
            'Seed': seed,
            'Feature_Set': feat_name,
            'Loss': m.get('loss', np.nan),
            'MAE': m.get('mae', np.nan),
            'RMSE': m.get('rmse', np.nan),
            'Euclidean_Mean': m.get('mean_euclidean', np.nan),
            'Euclidean_Std': m.get('std_euclidean', np.nan),
        }
        if feat_name in tim:
            row['Training_Time_Sec'] = tim[feat_name]['training_time_sec']
            row['Prediction_Time_Sec'] = tim[feat_name]['prediction_time_sec']
        rows.append(row)

    if rows:
        path = os.path.join(runs_dir, f'{model_type}_run{run_num:03d}_seed{seed}.csv')
        pd.DataFrame(rows).to_csv(path, index=False)


def save_aggregated_results(aggregated_data: dict, results_dir: str, model_type: str):
    """Save aggregated + per-run CSVs."""
    import pandas as pd
    agg_dir = os.path.join(results_dir, 'aggregated')
    runs_dir = os.path.join(results_dir, 'runs')
    os.makedirs(agg_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    # Aggregated CSV
    agg_results = aggregated_data['aggregated_results']
    df_agg = pd.DataFrame([
        {
            'Feature_Set': fn,
            'Loss_Mean': r['loss_mean'],
            'Loss_Std': r['loss_std'],
            'MAE_Mean': r['mae_mean'],
            'MAE_Std': r['mae_std'],
            'RMSE_Mean': r['rmse_mean'],
            'RMSE_Std': r['rmse_std'],
            'Euclidean_Mean': r['euclidean_mean'],
            'Euclidean_Std': r['euclidean_std'],
            'Training_Time_Mean': r['training_time_mean'],
            'Training_Time_Std': r['training_time_std'],
            'Prediction_Time_Mean': r['prediction_time_mean'],
            'Prediction_Time_Std': r['prediction_time_std'],
            'N_Params_Mean': r.get('n_params_mean', np.nan),
            'N_Runs': r['n_runs'],
        }
        for fn, r in agg_results.items()
    ])
    agg_path = os.path.join(agg_dir, f'{model_type}_aggregated_results.csv')
    df_agg.to_csv(agg_path, index=False)

    # All-runs CSV
    rows = []
    for run_idx, run_res in enumerate(aggregated_data['all_runs_results']):
        for fn, m in run_res.items():
            if isinstance(m, dict):
                rows.append({
                    'Run': run_idx + 1,
                    'Feature_Set': fn,
                    'Loss': m.get('loss', np.nan),
                    'MAE': m.get('mae', np.nan),
                    'RMSE': m.get('rmse', np.nan),
                    'Euclidean_Mean': m.get('mean_euclidean', np.nan),
                    'Euclidean_Std': m.get('std_euclidean', np.nan),
                    'N_Params': m.get('n_params', np.nan),
                })
    runs_path = os.path.join(runs_dir, f'{model_type}_all_runs_results.csv')
    pd.DataFrame(rows).to_csv(runs_path, index=False)

    print(f'\n💾 Results saved ({model_type}):')
    print(f'   Aggregated -> {agg_path}')
    print(f'   Per-run    -> {runs_path}')