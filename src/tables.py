"""
LaTeX table generation functions
"""

import os
import pandas as pd
import glob

def load_saved_results(results_base_dir="results_InLoQ"):
    """Load all saved aggregated results from results_InLoQ/"""
    model_data = {}
    
    model_configs = {
        'Inloq': 'Inloq',
        'Classical_ANN': 'Classical ANN', 
        'LSTM': 'LSTM',
        'GRU': 'GRU',
        'BiLSTM': 'BiLSTM',
    }
    
    for dir_name, display_name in model_configs.items():
        agg_path = os.path.join(results_base_dir, dir_name, 'aggregated', '*_aggregated_results.csv')
        files = glob.glob(agg_path)
        
        if files:
            df = pd.read_csv(files[0])
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
            
            model_data[display_name] = {'aggregated_results': aggregated_results}
            print(f"✅ Loaded {display_name} results")
        else:
            print(f"⚠️ No results found for {display_name}")
    
    return model_data


def save_latex_table(model_data: dict, results_dir: str):
    """Write a ready-to-paste LaTeX comparison table with per-column highlighting."""
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for model_name, data in model_data.items():
        for feat_name, agg in data['aggregated_results'].items():
            rows.append({
                'model': model_name,
                'feature_set': feat_name,
                'mae': agg['mae_mean'],
                'mae_std': agg['mae_std'],
                'rmse': agg['rmse_mean'],
                'rmse_std': agg['rmse_std'],
                'euc': agg['euclidean_mean'],
                'euc_std': agg['euclidean_std'],
                'train_s': agg['training_time_mean'],
                'n_runs': agg['n_runs'],
            })
    df = pd.DataFrame(rows)

    best_mae = set(df.groupby('feature_set')['mae'].idxmin().values)
    best_rmse = set(df.groupby('feature_set')['rmse'].idxmin().values)
    best_euc = set(df.groupby('feature_set')['euc'].idxmin().values)

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Localisation accuracy comparison (mean $\pm$ std over '
        + str(int(df['n_runs'].iloc[0])) + r' runs)}',
        r'\label{tab:comparison}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{llcccc}', r'\toprule',
        r'Model & Feature Set & MAE (m) & RMSE (m) & '
        r'Euclidean (m) & Train (s) \\',
        r'\midrule',
    ]

    for i, r in df.iterrows():
        mae_val = f"{r['mae']:.4f}$\\pm${r['mae_std']:.4f}"
        rmse_val = f"{r['rmse']:.4f}$\\pm${r['rmse_std']:.4f}"
        euc_val = f"{r['euc']:.4f}$\\pm${r['euc_std']:.4f}"

        if i in best_mae:
            mae_val = r'\textbf{\boldmath ' + mae_val + '}'
        if i in best_rmse:
            rmse_val = r'\textbf{\boldmath ' + rmse_val + '}'
        if i in best_euc:
            euc_val = r'\textbf{\boldmath ' + euc_val + '}'

        lines.append(
            f"{r['model']} & {r['feature_set']} & "
            f"{mae_val} & {rmse_val} & {euc_val} & {r['train_s']:.1f} \\\\"
        )

    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']

    path = os.path.join(results_dir, 'comparison_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'LaTeX table generated with per-column highlighting -> {path}')


def save_latex_table_best_marked(model_data: dict, results_dir: str):
    """Write a ready-to-paste LaTeX comparison table with best values in bold."""
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for model_name, data in model_data.items():
        if not data.get('aggregated_results'):
            continue
        for feat_name, agg in data['aggregated_results'].items():
            rows.append({
                'model': model_name,
                'feature_set': feat_name,
                'mae': agg['mae_mean'],
                'mae_std': agg['mae_std'],
                'rmse': agg['rmse_mean'],
                'rmse_std': agg['rmse_std'],
                'euc': agg['euclidean_mean'],
                'euc_std': agg['euclidean_std'],
                'train_s': agg['training_time_mean'],
                'n_runs': agg['n_runs'],
            })
    
    if not rows:
        print("⚠️ No data available to generate LaTeX table.")
        return

    df = pd.DataFrame(rows)

    df['is_best_mae'] = df.groupby('feature_set')['mae'].transform(lambda x: x == x.min())
    df['is_best_rmse'] = df.groupby('feature_set')['rmse'].transform(lambda x: x == x.min())
    df['is_best_euc'] = df.groupby('feature_set')['euc'].transform(lambda x: x == x.min())

    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{Localisation accuracy comparison (mean $\pm$ std over '
        + str(int(df['n_runs'].iloc[0])) + r' runs). \textbf{Bold} indicates best performance per feature set.}',
        r'\label{tab:comparison}',
        r'\resizebox{\linewidth}{!}{',
        r'\begin{tabular}{llcccc}', r'\toprule',
        r'Model & Feature Set & MAE (m) & RMSE (m) & Euclidean (m) & Train (s) \\',
        r'\midrule',
    ]

    for _, r in df.iterrows():
        mae_s = f"{r['mae']:.4f}$\\pm${r['mae_std']:.4f}"
        rmse_s = f"{r['rmse']:.4f}$\\pm${r['rmse_std']:.4f}"
        euc_s = f"{r['euc']:.4f}$\\pm${r['euc_std']:.4f}"

        if r['is_best_mae']:
            mae_s = r'\textbf{' + mae_s + '}'
        if r['is_best_rmse']:
            rmse_s = r'\textbf{' + rmse_s + '}'
        if r['is_best_euc']:
            euc_s = r'\textbf{' + euc_s + '}'

        lines.append(
            f"{r['model']} & {r['feature_set']} & "
            f"{mae_s} & {rmse_s} & {euc_s} & {r['train_s']:.1f} \\\\"
        )

    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']

    path = os.path.join(results_dir, 'comparison_table_best_bold.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'✅ LaTeX table (Multi-column Bold) -> {path}')