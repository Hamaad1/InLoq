import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.image as mpimg
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PALETTE = {
    'InLoq (Proposed)': '#1f77b4',
    'Classical ANN': '#ff7f0e',
    'LSTM': '#2ca02c',
    'GRU': '#d62728',
    'BiLSTM': '#9467bd',
    'InLoq (quantum only)': '#aec7e8',
}
FEATURE_LABELS = {
    "BT only"     : r"$B_t$",
    "Bx, By, Bz"  : r"$B_x, B_y, B_z$",
    "Bz, Bxy, BT" : r"$B_z, B_{xy}, B_t$",
}

# PLOT_PALETTE = {
#     'InLoq (Proposed)'  : '#1f77b4',
#     'Classical ANN'     : '#ff7f0e',
#     'LSTM'              : '#2ca02c',
#     'GRU'               : '#d62728',
#     'BiLSTM'            : '#9467bd',
#     'InLoq (quantum only)': '#aec7e8',
# }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def pad_sequence(seq_list, max_len=None):
    """Pad sequences to same length."""
    if max_len is None:
        max_len = max(len(x) for x in seq_list)
    padded = []
    for seq in seq_list:
        if len(seq) < max_len:
            seq = list(seq) + [seq[-1]] * (max_len - len(seq))
        padded.append(seq)
    return np.array(padded)

def pad_and_stack(lst, pad_val=np.nan):
    """Pad and stack sequences."""
    max_len = max(len(x) for x in lst)
    return np.vstack([np.pad(x, (0, max_len - len(x)), constant_values=pad_val) for x in lst])

def get_feature_label(fs):
    """Get formatted feature label."""
    return FEATURE_LABELS.get(fs, fs)

def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def collect_errors_over_runs(all_errors):
    """Collect errors across all runs for each feature set."""
    if not all_errors:
        return {}
    feature_sets = all_errors[0].keys()
    return {fs: np.concatenate([run[fs] for run in all_errors]) for fs in feature_sets}

def average_test_data_over_runs(all_test_data):
    """Average test predictions across runs."""
    feature_sets = all_test_data[0].keys()
    averaged = {}
    for fs in feature_sets:
        y_test = all_test_data[0][fs]['y_test']
        preds = np.stack([run[fs]['y_pred'] for run in all_test_data], axis=0)
        y_pred_mean = preds.mean(axis=0)
        averaged[fs] = {"y_test": y_test, "y_pred": y_pred_mean}
    return averaged

# ============================================================================
# PER-MODEL ANALYSIS PLOTS
# ============================================================================

def plot_training_history_with_uncertainty(all_histories, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    feature_sets = list(all_histories[0].keys())
    for feature_name in feature_sets:
        all_train_loss, all_val_loss, all_train_mae, all_val_mae = [], [], [], []
        for run_histories in all_histories:
            history = run_histories[feature_name]
            all_train_loss.append(history.history['loss'])
            all_val_loss.append(history.history['val_loss'])
            all_train_mae.append(history.history['mae'])
            all_val_mae.append(history.history['val_mae'])
        max_epochs = max(len(x) for x in all_train_loss)

        train_loss_array = pad_sequence(all_train_loss, max_epochs)
        val_loss_array   = pad_sequence(all_val_loss,   max_epochs)
        train_mae_array  = pad_sequence(all_train_mae,  max_epochs)
        val_mae_array    = pad_sequence(all_val_mae,    max_epochs)
        train_loss_mean, train_loss_std = train_loss_array.mean(0), train_loss_array.std(0)
        val_loss_mean,   val_loss_std   = val_loss_array.mean(0),   val_loss_array.std(0)
        train_mae_mean,  train_mae_std  = train_mae_array.mean(0),  train_mae_array.std(0)
        val_mae_mean,    val_mae_std    = val_mae_array.mean(0),    val_mae_array.std(0)
        epochs = range(1, max_epochs + 1)
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(epochs, train_loss_mean, label='Training Loss', color='blue')
        axes[0].fill_between(epochs, train_loss_mean - train_loss_std,
                             train_loss_mean + train_loss_std, alpha=0.2, color='blue')
        axes[0].plot(epochs, val_loss_mean, label='Validation Loss', color='orange')
        axes[0].fill_between(epochs, val_loss_mean - val_loss_std,
                             val_loss_mean + val_loss_std, alpha=0.2, color='orange')
        axes[0].set_title(f'{model_name} - {get_feature_label(feature_name)} - Loss (Mean ± Std)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MAE Loss')
        axes[0].legend()
        axes[0].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        axes[1].plot(epochs, train_mae_mean, label='Training MAE', color='blue')
        axes[1].fill_between(epochs, train_mae_mean - train_mae_std,
                             train_mae_mean + train_mae_std, alpha=0.2, color='blue')
        axes[1].plot(epochs, val_mae_mean, label='Validation MAE', color='orange')
        axes[1].fill_between(epochs, val_mae_mean - val_mae_std,
                             val_mae_mean + val_mae_std, alpha=0.2, color='orange')
        axes[1].set_title(f'{model_name} - {get_feature_label(feature_name)} - MAE (Mean ± Std)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE Loss')
        axes[1].legend()
        axes[1].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{feature_name}_training_uncertainty.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ Uncertainty plot saved for {feature_name}")


def plot_all_runs_overlay(all_histories, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    feature_sets = list(all_histories[0].keys())
    for feature_name in feature_sets:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        all_val_mae = []
        for run_histories in all_histories:
            history = run_histories[feature_name]
            epochs  = range(1, len(history.history['loss']) + 1)
            axes[0].plot(epochs, history.history['loss'],     alpha=0.3, color='blue',   linewidth=1)
            axes[0].plot(epochs, history.history['val_loss'], alpha=0.3, color='orange', linewidth=1)
            axes[1].plot(epochs, history.history['mae'],      alpha=0.3, color='blue',   linewidth=1)
            axes[1].plot(epochs, history.history['val_mae'],  alpha=0.3, color='orange', linewidth=1)
            all_val_mae.append(history.history['val_mae'])
        max_epochs = max(len(x) for x in all_val_mae)

        def pad_and_mean(data_list):
            padded = []
            for data in data_list:
                if len(data) < max_epochs:
                    data = list(data) + [data[-1]] * (max_epochs - len(data))
                padded.append(data)
            return np.mean(padded, axis=0)

        all_train_loss = [h[feature_name].history['loss']     for h in all_histories]
        all_val_loss   = [h[feature_name].history['val_loss'] for h in all_histories]
        all_train_mae  = [h[feature_name].history['mae']      for h in all_histories]
        mean_train_loss = pad_and_mean(all_train_loss)
        mean_val_loss   = pad_and_mean(all_val_loss)
        mean_train_mae  = pad_and_mean(all_train_mae)
        mean_val_mae    = pad_and_mean(all_val_mae)
        mean_epochs     = range(1, max_epochs + 1)
        axes[0].plot(mean_epochs, mean_train_loss, color='blue',   linewidth=2.5, label='Mean Training Loss')
        axes[0].plot(mean_epochs, mean_val_loss,   color='orange', linewidth=2.5, label='Mean Validation Loss')
        axes[0].set_title(f'{model_name} - {get_feature_label(feature_name)} - Loss (All Runs)')
        axes[0].set_xlabel('Epoch')
        axes[0].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(mean_epochs, mean_train_mae, color='blue',   linewidth=2.5, label='Mean Training MAE')
        axes[1].plot(mean_epochs, mean_val_mae,   color='orange', linewidth=2.5, label='Mean Validation MAE')
        axes[1].set_title(f'{model_name} - {get_feature_label(feature_name)} - MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{feature_name}_all_runs_overlay.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ Overlay plot saved for {feature_name}")


# 

def plot_metrics_with_error_bars(aggregated_results, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    feature_sets   = list(aggregated_results.keys())
    feature_labels = [get_feature_label(fs) for fs in feature_sets]
    metrics = {
        'MAE'            : ('mae_mean',       'mae_std'),
        'RMSE'           : ('rmse_mean',      'rmse_std'),
        'Euclidean Error': ('euclidean_mean', 'euclidean_std'),
    }
    x     = np.arange(len(feature_sets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (metric_name, (mean_key, std_key)) in enumerate(metrics.items()):
        means = [aggregated_results[fs][mean_key] for fs in feature_sets]
        stds  = [aggregated_results[fs][std_key]  for fs in feature_sets]
        ax.bar(x + idx * width, means, width, yerr=stds, label=metric_name, capsize=5, alpha=0.8)
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('Error (meters)')
    ax.set_title(f'{model_name} - Aggregated Metrics (Mean ± Std)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_labels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_metrics_error_bars.png'), bbox_inches='tight')
    plt.close()


def plot_convergence_analysis(all_histories, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    feature_sets = list(all_histories[0].keys())
    thresholds   = [1.0, 2.0, 3.0, 4.0, 5.0]
    for feature_name in feature_sets:
        label = get_feature_label(feature_name)
        convergence_data = {t: [] for t in thresholds}
        for run_histories in all_histories:
            val_mae = run_histories[feature_name].history['val_mae']
            for threshold in thresholds:
                epochs_to_converge = next(
                    (epoch for epoch, mae in enumerate(val_mae, 1) if mae <= threshold),
                    len(val_mae))
                convergence_data[threshold].append(epochs_to_converge)
        means = [np.mean(convergence_data[t]) for t in thresholds]
        stds  = [np.std(convergence_data[t])  for t in thresholds]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(thresholds, means, yerr=stds, fmt='o-',
                    ecolor='red', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('MAE Threshold (meters)')
        ax.set_ylabel('Epochs to Converge (Mean ± Std)')
        ax.set_title(f'{model_name} - {label} - Convergence Analysis')
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{feature_name}_convergence.png'), bbox_inches='tight')
        plt.close()
        print(f"✅ Convergence plot saved for {label}")


def plot_run_to_run_variability(all_results, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    feature_sets = list(all_results[0].keys())
    n_runs       = len(all_results)
    for metric_name in ['mae', 'rmse', 'mean_euclidean']:
        fig, ax = plt.subplots(figsize=(12, 6))
        for feature_name in feature_sets:
            label = get_feature_label(feature_name)
            metric_values = [all_results[i][feature_name][metric_name] for i in range(n_runs)]
            runs = range(1, n_runs + 1)
            ax.plot(runs, metric_values, marker='o', linewidth=2, markersize=6, label=label)
        ax.set_xlabel('Run')
        ax.set_ylabel(f'{metric_name.upper()} (meters)')
        ax.set_title(f'{model_name} - {metric_name.upper()} Variability Across Runs')
        ax.set_xticks(runs)
        ax.legend(title="Feature Set")
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'{model_name}_{metric_name}_variability.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Run-to-run variability plot saved: {save_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_multirun_analysis(aggregated_data, results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📊 Running multi-run analysis for {model_name}...")
    all_histories      = aggregated_data['all_histories']
    all_errors         = aggregated_data['all_errors']
    all_results        = aggregated_data['all_runs_results']
    aggregated_results = aggregated_data['aggregated_results']
    
    print("  📈 Plotting training history with uncertainty...")
    plot_training_history_with_uncertainty(all_histories, results_dir, model_name)
   
    print("  📈 Plotting all runs overlay...")
    plot_all_runs_overlay(all_histories, results_dir, model_name)
    
    # print("  📊 Plotting error distributions...")
    # plot_error_distribution_across_runs(all_errors, results_dir, model_name)
    
    print("  📊 Plotting metrics with error bars...")
    plot_metrics_with_error_bars(aggregated_results, results_dir, model_name)
    
    print("  📈 Plotting convergence analysis...")
    plot_convergence_analysis(all_histories, results_dir, model_name)
   
    print("  📊 Plotting run-to-run variability...")
    plot_run_to_run_variability(all_results, results_dir, model_name)
    
    print(f"✅ Multi-run analysis complete for {model_name}!")


# ============================================================================
# SPATIAL PREDICTION PLOTS (CRITICAL - WAS MISSING)
# ============================================================================

def analyze_predictions_multirun(test_data, results_dir, img_path, model_name): #analyze_predictions_per_reference_point
    """
    Generate spatial prediction plots with per-reference point aggregation.
    
    Args:
        test_data: Dictionary with model_name -> list of runs, each containing y_test and y_pred
        results_dir: Directory to save plots
        img_path: Path to building floorplan image
        model_name: Name of the model
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Load floorplan image
    try:
        img = mpimg.imread(img_path)
    except:
        img = None
        print(f"⚠️ Could not load image from {img_path}, using white background")
    
    # Get map boundaries
    x_min, x_max = 0, 45
    y_min, y_max = 0, 12
    
    # Collect all runs data
    runs_data = test_data[model_name]
    
    # Group by unique reference point (round to 2 decimals for grouping)
    ref_points = defaultdict(list)
    
    for run_idx, run in enumerate(runs_data):
        y_test = run['y_test']
        y_pred = run['y_pred']
        
        # Round to avoid floating point precision issues
        rounded_coords = np.round(y_test, decimals=3)
        
        for i in range(len(y_test)):
            ref_key = tuple(rounded_coords[i])
            ref_points[ref_key].append({
                'y_true': y_test[i],
                'y_pred': y_pred[i],
                'run_idx': run_idx
            })
    
    # =========================================================
    # 1. CLEAN MEAN PREDICTION PER REFERENCE POINT
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    if img is not None:
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
    
    # Collect means for plotting
    ref_points_list = []
    mean_predictions = []
    std_predictions = []
    errors = []
    
    for ref_key, predictions in ref_points.items():
        true_x, true_y = ref_key
        pred_x = np.mean([p['y_pred'][0] for p in predictions])
        pred_y = np.mean([p['y_pred'][1] for p in predictions])
        std_x = np.std([p['y_pred'][0] for p in predictions])
        std_y = np.std([p['y_pred'][1] for p in predictions])
        error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        
        ref_points_list.append((true_x, true_y))
        mean_predictions.append((pred_x, pred_y))
        std_predictions.append((std_x, std_y))
        errors.append(error)
        
        # Plot mean prediction
        ax.scatter(pred_x, pred_y, s=80, c='green', marker='x', 
                   linewidths=2, zorder=4, label='Prediction' if len(ref_points_list) == 1 else "")
        
        # Plot true location
        ax.scatter(true_x, true_y, s=80, c='red', marker='o', 
                   alpha=0.7, edgecolors='black', zorder=3, 
                   label='True Loc' if len(ref_points_list) == 1 else "")
        
        # Draw error line
        ax.plot([true_x, pred_x], [true_y, pred_y], 'b-', alpha=0.4, linewidth=1)
    
    ax.set_title(f"{model_name} - Mean Prediction per Reference Point", fontsize=14)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{model_name}_mean_per_reference.png"), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # =========================================================
    # 2. MEAN WITH ERROR BARS (CONFIDENCE ELLIPSES)
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    if img is not None:
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
    
    for (true_x, true_y), (pred_x, pred_y), (std_x, std_y) in zip(
            ref_points_list, mean_predictions, std_predictions):
        
        # Plot true location
        ax.scatter(true_x, true_y, s=80, c='red', marker='o', alpha=0.7, edgecolors='black')
        
        # Plot mean prediction
        ax.scatter(pred_x, pred_y, s=80, c='green', marker='x', linewidths=2)
        
        # Draw error ellipse (2 standard deviations)
        from matplotlib.patches import Ellipse
        angle = 0
        width = 2 * std_x * 2  # 2 standard deviations
        height = 2 * std_y * 2
        ellipse = Ellipse(xy=(pred_x, pred_y), width=width, height=height,
                         angle=angle, facecolor='green', alpha=0.2, edgecolor='green')
        ax.add_patch(ellipse)
        
        # Error line
        ax.plot([true_x, pred_x], [true_y, pred_y], 'b-', alpha=0.5, linewidth=1)
    
    ax.set_title(f"{model_name} - Prediction with 2σ Error Ellipses", fontsize=14)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{model_name}_error_ellipses.png"), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # =========================================================
    # 3. ERROR HEATMAP PER REFERENCE POINT
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    if img is not None:
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
    
    # Create scatter with color representing error magnitude
    sc = ax.scatter([p[0] for p in ref_points_list], 
                    [p[1] for p in ref_points_list],
                    c=errors, s=100, cmap='RdYlGn_r', vmin=0, vmax=np.percentile(errors, 95),
                    edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05, aspect=30)
    # Add common colorbar with better positioning
    # cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.05, aspect=30)
    # cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Mean Euclidean Error (m)', fontsize=12)
    
    # Add error labels for large errors
    for (true_x, true_y), error in zip(ref_points_list, errors):
        if error > np.percentile(errors, 80):  # Show only top 20% errors
            ax.annotate(f'{error:.2f}m', xy=(true_x, true_y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, color='black',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_title(f"{model_name} - Error Heatmap per Reference Point", fontsize=14)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{model_name}_error_heatmap.png"), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # =========================================================
    # 4. ERROR DISTRIBUTION PLOTS
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    axes[0].hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(errors):.2f}m')
    axes[0].axvline(np.median(errors), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(errors):.2f}m')
    axes[0].set_xlabel('Euclidean Error (m)', fontsize=12)
    axes[0].set_ylabel('Number of Reference Points', fontsize=12)
    axes[0].set_title(f'{model_name} - Error Distribution per Reference Point', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot with percentiles
    box = axes[1].boxplot(errors, patch_artist=True, widths=0.5)
    box['boxes'][0].set_facecolor('lightblue')
    axes[1].set_ylabel('Euclidean Error (m)', fontsize=12)
    axes[1].set_xticklabels([model_name])
    axes[1].set_title(f'{model_name} - Error Statistics\n(n={len(errors)} reference points)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotations
    stats_text = f'Mean: {np.mean(errors):.2f}m\nMedian: {np.median(errors):.2f}m\nStd: {np.std(errors):.2f}m\n95%: {np.percentile(errors, 95):.2f}m'
    axes[1].text(1.1, np.percentile(errors, 95), stats_text, 
                transform=axes[1].transData, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{model_name}_error_distribution.png"), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # =========================================================
    # 5. PREDICTION vs TRUE SCATTER PLOT (Per Reference Point)
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # X coordinate comparison
    true_x = [p[0] for p in ref_points_list]
    pred_x = [mp[0] for mp in mean_predictions]
    
    axes[0].scatter(true_x, pred_x, alpha=0.6, s=60, c='blue', edgecolors='black')
    axes[0].plot([0, 45], [0, 45], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('True X (m)', fontsize=12)
    axes[0].set_ylabel('Predicted X (m)', fontsize=12)
    axes[0].set_title(f'{model_name} - X Coordinate Prediction', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add R² annotation
    from sklearn.metrics import r2_score
    r2_x = r2_score(true_x, pred_x)
    axes[0].annotate(f'R² = {r2_x:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Y coordinate comparison
    true_y = [p[1] for p in ref_points_list]
    pred_y = [mp[1] for mp in mean_predictions]
    
    axes[1].scatter(true_y, pred_y, alpha=0.6, s=60, c='green', edgecolors='black')
    axes[1].plot([0, 12], [0, 12], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Y (m)', fontsize=12)
    axes[1].set_ylabel('Predicted Y (m)', fontsize=12)
    axes[1].set_title(f'{model_name} - Y Coordinate Prediction', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    r2_y = r2_score(true_y, pred_y)
    axes[1].annotate(f'R² = {r2_y:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{model_name}_prediction_scatter.png"), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    
    # Save statistics to CSV
    import pandas as pd
    stats_df = pd.DataFrame({
        'Reference_X': true_x,
        'Reference_Y': true_y,
        'Predicted_X_Mean': pred_x,
        'Predicted_Y_Mean': pred_y,
        'Std_X': [s[0] for s in std_predictions],
        'Std_Y': [s[1] for s in std_predictions],
        'Mean_Error_m': errors
    })
    stats_df.to_csv(os.path.join(results_dir, f"{model_name}_per_reference_stats.csv"), index=False)
    
    print(f"✅ Saved per-reference point analysis for {model_name}")
    print(f"   - {len(ref_points_list)} unique reference points")
    print(f"   - Mean error: {np.mean(errors):.2f}m ± {np.std(errors):.2f}m")
    
    return ref_points_list, mean_predictions, errors


def plot_mean_comparison_overlay(qnn_data, ann_data, results_dir, img_path):
    """Overlays QNN vs ANN mean predictions for EVERY feature set."""
    os.makedirs(results_dir, exist_ok=True)
    img = mpimg.imread(img_path)
    feature_sets = qnn_data['all_test_data'][0].keys()
    
    for fs in feature_sets:
        y_test = qnn_data['all_test_data'][0][fs]['y_test']
        q_mean = np.array([run[fs]['y_pred'] for run in qnn_data['all_test_data']]).mean(axis=0)
        a_mean = np.array([run[fs]['y_pred'] for run in ann_data['all_test_data']]).mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img, extent=[0, 45, 0, 12])
        ax.scatter(y_test[:, 0], y_test[:, 1], c='red', label='True Location', 
                   s=50, edgecolors='black', zorder=3)
        ax.scatter(q_mean[:, 0], q_mean[:, 1], c='cyan', marker='P', 
                   label='Inloq (Mean)', s=60, edgecolors='black', zorder=5)
        ax.scatter(a_mean[:, 0], a_mean[:, 1], c='yellow', marker='X', 
                   label='Classical ANN (Mean)', s=60, edgecolors='black', zorder=4)
        ax.set_title(f"Spatial Accuracy Comparison ({fs})")
        ax.set_xlabel("Room Length (m)")
        ax.set_ylabel("Room Width (m)")
        ax.legend(loc='upper right', frameon=True)
        save_path = os.path.join(results_dir, f"mean_comparison_overlay_{fs}.png")
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"✅ Overlay saved for {fs} -> {save_path}")


# ============================================================================
# QNN vs ANN PAIR PLOTS
# ============================================================================

def plot_combined_training_history_multi(qnn_histories_list, ann_histories_list, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    feature_sets = qnn_histories_list[0].keys()
    for feature_name in feature_sets:
        qnn_losses     = [h[feature_name].history['loss']     for h in qnn_histories_list]
        qnn_val_losses = [h[feature_name].history['val_loss'] for h in qnn_histories_list]
        ann_losses     = [h[feature_name].history['loss']     for h in ann_histories_list]
        ann_val_losses = [h[feature_name].history['val_loss'] for h in ann_histories_list]
        qnn_loss_arr = pad_and_stack(qnn_losses)
        qnn_val_arr  = pad_and_stack(qnn_val_losses)
        ann_loss_arr = pad_and_stack(ann_losses)
        ann_val_arr  = pad_and_stack(ann_val_losses)
        qnn_loss_mean, qnn_loss_std = np.nanmean(qnn_loss_arr, 0), np.nanstd(qnn_loss_arr, 0)
        qnn_val_mean,  qnn_val_std  = np.nanmean(qnn_val_arr,  0), np.nanstd(qnn_val_arr,  0)
        ann_loss_mean, ann_loss_std = np.nanmean(ann_loss_arr, 0), np.nanstd(ann_loss_arr, 0)
        ann_val_mean,  ann_val_std  = np.nanmean(ann_val_arr,  0), np.nanstd(ann_val_arr,  0)
        plt.figure(figsize=(10, 6))
        epochs = np.arange(len(qnn_loss_mean))
        plt.plot(epochs, qnn_loss_mean, label="Inloq Train", color="blue")
        plt.fill_between(epochs, qnn_loss_mean-qnn_loss_std, qnn_loss_mean+qnn_loss_std, alpha=0.2, color="blue")
        plt.plot(epochs, qnn_val_mean, label="Inloq Val", color="cyan")
        plt.fill_between(epochs, qnn_val_mean-qnn_val_std, qnn_val_mean+qnn_val_std, alpha=0.2, color="cyan")
        plt.plot(epochs, ann_loss_mean, label="Classical ANN Train", color="red")
        plt.fill_between(epochs, ann_loss_mean-ann_loss_std, ann_loss_mean+ann_loss_std, alpha=0.2, color="red")
        plt.plot(epochs, ann_val_mean, label="Classical ANN Val", color="orange")
        plt.fill_between(epochs, ann_val_mean-ann_val_std, ann_val_mean+ann_val_std, alpha=0.2, color="orange")
        feature_label = get_feature_label(feature_name)
        plt.xlabel("Epochs")
        plt.ylabel("MAE Loss")
        plt.title(f"Training History (avg over runs) - {feature_label}")
        plt.legend()
        plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.tight_layout()
        save_path = os.path.join(results_dir, f"training_history_multi_{feature_name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Combined training history saved for {feature_label}: {save_path}")




def plot_predictions_vs_true(qnn_test_data, ann_test_data, comparison_dir):
    os.makedirs(comparison_dir, exist_ok=True)
    for fs in qnn_test_data.keys():
        if fs not in ann_test_data:
            continue
        y_test_qnn = qnn_test_data[fs]['y_test']
        y_pred_qnn = qnn_test_data[fs]['y_pred']
        y_test_ann = ann_test_data[fs]['y_test']
        y_pred_ann = ann_test_data[fs]['y_pred']
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].scatter(y_test_qnn[:, 0], y_test_qnn[:, 1], label="True", alpha=0.5, s=20, color='red')
        axes[0].scatter(y_pred_qnn[:, 0], y_pred_qnn[:, 1], label="Pred", alpha=0.5, s=30, color='green')
        axes[0].set_title(f"Inloq - Feature Set: {get_feature_label(fs)}")
        axes[0].set_xlabel("X Coordinate")
        axes[0].set_ylabel("Y Coordinate")
        axes[0].legend()
        axes[1].scatter(y_test_ann[:, 0], y_test_ann[:, 1], label="True", alpha=0.5, s=20, color='red')
        axes[1].scatter(y_pred_ann[:, 0], y_pred_ann[:, 1], label="Pred", alpha=0.5, s=30, color='blue')
        axes[1].set_title(f"Classical ANN - Feature Set: {get_feature_label(fs)}")
        axes[1].set_xlabel("X Coordinate")
        axes[1].set_ylabel("Y Coordinate")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f"pred_vs_true_{fs}.png"), bbox_inches="tight")
        plt.close()


def save_aggregate_test_results(test_data, results_dir):
    """Save aggregated test results for spatial predictions."""
    agg_dir  = os.path.join(results_dir, "aggregated")
    runs_dir = os.path.join(results_dir, "runs")
    os.makedirs(agg_dir,  exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    aggregate_rows = []
    for model_name, runs in test_data.items():
        all_errors = []
        for idx, run in enumerate(runs):
            y_test = run['y_test']
            y_pred = run['y_pred']
            errors = np.linalg.norm(y_test - y_pred, axis=1)
            all_errors.append(errors)
            df_pred = pd.DataFrame({
                "y_true_x"       : y_test[:, 0],
                "y_true_y"       : y_test[:, 1],
                "y_pred_x"       : y_pred[:, 0],
                "y_pred_y"       : y_pred[:, 1],
                "euclidean_error": errors,
            })
            df_pred.to_csv(os.path.join(runs_dir, f"{model_name}_run{idx+1}_predictions.csv"), index=False)
        all_errors = np.array(all_errors)
        mean_error_per_sample = all_errors.mean(axis=0)
        aggregate_rows.append({
            "Model"         : model_name,
            "Mean_Euclidean": mean_error_per_sample.mean(),
            "Std_Euclidean" : mean_error_per_sample.std(),
            "Max_Euclidean" : mean_error_per_sample.max(),
            "Min_Euclidean" : mean_error_per_sample.min(),
        })
    df_agg = pd.DataFrame(aggregate_rows)
    df_agg.to_csv(os.path.join(agg_dir, "aggregate_test_results.csv"), index=False)
    print(f"\n💾 Test results saved:")
    print(f"   - Aggregated: {agg_dir}/aggregate_test_results.csv")
    print(f"   - Per-run   : {runs_dir}/<model>_runX_predictions.csv")


# ============================================================================
# run_multirun_comparison_1 (Spatial analysis for QNN vs ANN)
# ============================================================================

def run_multirun_comparison_1(qnn_multi_run_data, ann_multi_run_data,
                              comparison_dir, img_path, extra_models=None):
    """
    Spatial analysis ONLY (per-reference point plots).
    Comparison plots (CDF, boxplots, error bars) are handled by run_all_models_comparison.
    """
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Collect all models
    all_models_to_plot = {'Inloq': qnn_multi_run_data, 'Classical ANN': ann_multi_run_data}
    if extra_models:
        all_models_to_plot.update(extra_models)
    
    # Get feature sets
    first_model = list(all_models_to_plot.values())[0]
    feature_sets = first_model['all_test_data'][0].keys()
    
    # =========================================================================
    # Per-feature set spatial analysis (NO comparison plots here)
    # =========================================================================
    for fs in feature_sets:
        print(f"\n📊 Processing feature set: {fs}")
        
        # Prepare test data
        current_fs_test_data = {}
        for model_name, model_data in all_models_to_plot.items():
            current_fs_test_data[model_name] = []
            for run in model_data['all_test_data']:
                current_fs_test_data[model_name].append({
                    'y_test': run[fs]['y_test'], 
                    'y_pred': run[fs]['y_pred']
                })
        
        # Create directory for this feature set
        fs_dir = os.path.join(comparison_dir, f"FeatureSet_{fs}")
        os.makedirs(fs_dir, exist_ok=True)
        
        # Generate per-reference point analysis for each model
         
        all_results = {}
        for model_name in current_fs_test_data.keys():
            print(f"   Analyzing {model_name}...")
            ref_points, predictions, errors = analyze_predictions_multirun(
                current_fs_test_data, fs_dir, img_path, model_name
            )
            all_results[model_name] = {
                'ref_points': ref_points,
                'predictions': predictions,
                'errors': errors
            }
        
        if len(all_models_to_plot) > 1:
            fig, axes = plt.subplots(1, len(all_models_to_plot), 
                                      figsize=(6*len(all_models_to_plot), 6))
            if len(all_models_to_plot) == 1:
                axes = [axes]
            
            try:
                img = mpimg.imread(img_path)
            except:
                img = None
            
            x_min, x_max = 0, 45
            y_min, y_max = 0, 12
            
            # Find global error range for consistent coloring
            all_errors = [e for r in all_results.values() for e in r['errors']]
            vmax = np.percentile(all_errors, 95) if all_errors else 1
            
            for idx, (model_name, results) in enumerate(all_results.items()):
                ax = axes[idx]
                if img is not None:
                    ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
                
                errors = results['errors']
                ref_points = results['ref_points']
                
                sc = ax.scatter([p[0] for p in ref_points], [p[1] for p in ref_points],
                              c=errors, s=80, cmap='RdYlGn_r', 
                              vmin=0, vmax=vmax,
                              edgecolors='black', linewidth=0.5)
                
                ax.set_title(f"{model_name}\nMean Error: {np.mean(errors):.2f}m", fontsize=12)
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.grid(True, alpha=0.3)
            
            # Add common colorbar
            # cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05, aspect=30)
            # cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.05, aspect=30)
            cbar.set_label('Mean Euclidean Error (m)', fontsize=10)
            
            plt.suptitle(f"Per-Reference Point Error Comparison - {fs}", fontsize=14)
            plt.tight_layout()
            fig.savefig(os.path.join(fs_dir, "comparison_error_heatmap.png"), bbox_inches="tight", dpi=300)
            plt.close(fig)
            
        # Save aggregate test results
        save_aggregate_test_results(current_fs_test_data, fs_dir)
    
    print(f"\n✅ All spatial analysis plots saved to {comparison_dir}")
    
def run_multirun_comparison_1_orig(qnn_multi_run_data, ann_multi_run_data,
                                        comparison_dir, img_path, extra_models=None):
    """
    Enhanced version that generates:
    1. Per-reference point analysis (heatmaps, error ellipses, distributions)
    2. CDF plots for all models
    3. Violin plots for all models
    4. Combined training history plots
    5. Aggregate test results
    """
    os.makedirs(comparison_dir, exist_ok=True)
    
    # =========================================================================
    # 1. Collect all models to plot
    # =========================================================================
    all_models_to_plot = {'Inloq': qnn_multi_run_data, 'Classical ANN': ann_multi_run_data}
    if extra_models:
        all_models_to_plot.update(extra_models)
    
    # Get feature sets from the first model's test data
    first_model = list(all_models_to_plot.values())[0]
    feature_sets = first_model['all_test_data'][0].keys()
    # feature_sets = qnn_multi_run_data['all_test_data'][0].keys() #original, remove above two line
    
    # =========================================================================
    # 2. Per-feature set analysis (includes per-reference point plots)
    # =========================================================================
    for fs in feature_sets:
        print(f"\n📊 Processing feature set: {fs}")
        
        # Prepare test data for this feature set across all models
        current_fs_test_data = {}
        for model_name, model_data in all_models_to_plot.items():
            current_fs_test_data[model_name] = []
            for run in model_data['all_test_data']:
                current_fs_test_data[model_name].append({
                    'y_test': run[fs]['y_test'], 
                    'y_pred': run[fs]['y_pred']
                })
        
        # Create directory for this feature set
        fs_dir = os.path.join(comparison_dir, f"FeatureSet_{fs}")
        os.makedirs(fs_dir, exist_ok=True)
        
        # Generate per-reference point analysis for each model
        all_results = {}
        for model_name in current_fs_test_data.keys():
            print(f"   Analyzing {model_name}...")
            ref_points, predictions, errors = analyze_predictions_multirun(
                current_fs_test_data, fs_dir, img_path, model_name
            )
            all_results[model_name] = {
                'ref_points': ref_points,
                'predictions': predictions,
                'errors': errors
            }
        #     # Generate per-reference point analysis for each model
        # for model_name in current_fs_test_data.keys():
        #     print(f"   Analyzing {model_name}...")
        #     analyze_predictions_multirun(
        #         current_fs_test_data, fs_dir, img_path, model_name
        #     )
        
        # save_aggregate_test_results(current_fs_test_data, fs_dir)
        
        # Create comparison error heatmap across models
        if len(all_models_to_plot) > 1:
            fig, axes = plt.subplots(1, len(all_models_to_plot), 
                                      figsize=(6*len(all_models_to_plot), 6))
            if len(all_models_to_plot) == 1:
                axes = [axes]
            
            try:
                img = mpimg.imread(img_path)
            except:
                img = None
            
            x_min, x_max = 0, 45
            y_min, y_max = 0, 12
            
            # Find global error range for consistent coloring
            all_errors = [e for r in all_results.values() for e in r['errors']]
            vmax = np.percentile(all_errors, 95) if all_errors else 1
            
            for idx, (model_name, results) in enumerate(all_results.items()):
                ax = axes[idx]
                if img is not None:
                    ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
                
                errors = results['errors']
                ref_points = results['ref_points']
                
                sc = ax.scatter([p[0] for p in ref_points], [p[1] for p in ref_points],
                              c=errors, s=80, cmap='RdYlGn_r', 
                              vmin=0, vmax=vmax,
                              edgecolors='black', linewidth=0.5)
                
                ax.set_title(f"{model_name}\nMean Error: {np.mean(errors):.2f}m", fontsize=12)
                ax.set_xlabel("X Position (m)")
                ax.set_ylabel("Y Position (m)")
                ax.grid(True, alpha=0.3)
            
            # Add common colorbar
            # cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.15, fraction=0.05, aspect=30)
            # cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.05, aspect=30)
            cbar.set_label('Mean Euclidean Error (m)', fontsize=10)
            
            plt.suptitle(f"Per-Reference Point Error Comparison - {fs}", fontsize=14)
            plt.tight_layout()
            fig.savefig(os.path.join(fs_dir, "comparison_error_heatmap.png"), bbox_inches="tight", dpi=300)
            plt.close(fig)
        
        # Save aggregate test results (original functionality)
        save_aggregate_test_results(current_fs_test_data, fs_dir)
    
    # =========================================================================
    # 3. Combined Training History Plots
    # =========================================================================
    print("\n📈 Generating combined training history plots...")
    try:
        plot_combined_training_history_multi(
            qnn_multi_run_data['all_histories'], 
            ann_multi_run_data['all_histories'], 
            comparison_dir,
            extra_models={k: v['all_histories'] for k, v in extra_models.items()} if extra_models else None
        )
    except Exception as e:
        print(f"   ⚠️ Could not generate training history plots: {e}")
    
    # =========================================================================
    # 4. CDF Plots (Cumulative Distribution Function)
    # =========================================================================
    print("\n📊 Generating CDF plots for all models...")
    try:
        # Collect errors for all models
        all_models_errors = {}
        
        # Add Inloq errors
        qnn_errors = collect_errors_over_runs(qnn_multi_run_data['all_errors'])
        all_models_errors['Inloq'] = qnn_errors
        
        # Add ANN errors
        ann_errors = collect_errors_over_runs(ann_multi_run_data['all_errors'])
        all_models_errors['Classical ANN'] = ann_errors
        
        # Add extra models
        if extra_models:
            for model_name, model_data in extra_models.items():
                model_errors = collect_errors_over_runs(model_data['all_errors'])
                all_models_errors[model_name] = model_errors
        
        # Generate CDF plots
        # plot_cdf_per_feature(all_models_errors, comparison_dir)
        
    except Exception as e:
        print(f"   ⚠️ Could not generate CDF plots: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 5. Violin Plots
    # =========================================================================
    # print("\n🎻 Generating violin plots for all models...")
    # try:
    #     plot_violin_per_feature(all_models_errors, comparison_dir)
    # except Exception as e:
    #     print(f"   ⚠️ Could not generate violin plots: {e}")
    #     traceback.print_exc()
    
    # =========================================================================
    # 6. Generate additional comparison plots
    # =========================================================================
    print("\n📈 Generating additional comparison plots...")
    
    # Plot error bars comparison
    try:
        plot_error_bars_comparison(all_models_errors, comparison_dir)
    except Exception as e:
        print(f"   ⚠️ Could not generate error bar plots: {e}")
    
    # Plot box plots comparison
    try:
        plot_boxplot_comparison(all_models_errors, comparison_dir)
    except Exception as e:
        print(f"   ⚠️ Could not generate boxplot comparison: {e}")
    
def plot_error_bars_comparison(all_models_errors, comparison_dir):
    """
    Generate error bar plots comparing models across feature sets.
    """
    if not all_models_errors:
        return
    
    # Get feature sets
    feature_sets = list(all_models_errors.values())[0].keys()
    models = list(all_models_errors.keys())
    
    # Prepare data
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_sets))
    width = 0.8 / len(models)
    
    for idx, model_name in enumerate(models):
        means = []
        stds = []
        for fs in feature_sets:
            errors = all_models_errors[model_name][fs]
            means.append(np.mean(errors))
            stds.append(np.std(errors))
        
        offset = (idx - len(models)/2) * width
        ax.bar(x + offset, means, width, label=model_name, 
               yerr=stds, capsize=3, alpha=0.7)
    
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('Mean Euclidean Error (m)')
    ax.set_title('Error Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(comparison_dir, 'error_bars_comparison.png'), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"   ✅ Saved error bar comparison to {comparison_dir}")


def plot_boxplot_comparison(all_models_errors, comparison_dir):
    """
    Generate box plots comparing models across feature sets.
    """
    if not all_models_errors:
        return
    
    feature_sets = list(all_models_errors.values())[0].keys()
    models = list(all_models_errors.keys())
    
    fig, axes = plt.subplots(1, len(feature_sets), 
                              figsize=(5*len(feature_sets), 6))
    if len(feature_sets) == 1:
        axes = [axes]
    
    for idx, fs in enumerate(feature_sets):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        for model_name in models:
            errors = all_models_errors[model_name][fs]
            data_to_plot.append(errors)
            labels.append(model_name)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightgray']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
        
        ax.set_title(f'Feature Set: {fs}')
        ax.set_ylabel('Euclidean Error (m)')
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Error Distribution Comparison Across Models', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(comparison_dir, 'boxplot_comparison.png'), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"   ✅ Saved boxplot comparison to {comparison_dir}")

    print(f"\n✅ All spatial analysis plots saved to {comparison_dir}")


# ============================================================================
# MULTI-MODEL COMPARISON HELPERS
# ============================================================================

def _avg_timings(model_data: dict) -> dict:
    """Return averaged timings for a model."""
    timings = {}
    if 'all_timings' not in model_data or not model_data['all_timings']:
        return timings
    for fs in model_data['all_timings'][0].keys():
        timings[fs] = {
            'training_time_sec': float(np.mean([r[fs].get('training_time_sec', 0) for r in model_data['all_timings']])),
            'prediction_time_sec': float(np.mean([r[fs].get('prediction_time_sec', 0) for r in model_data['all_timings']])),
        }
    return timings


def _plot_multimodel_bar(all_models: dict, save_dir: str):
    """Grouped bar chart for all models."""
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(all_models.keys())
    feature_sets = list(next(iter(all_models.values()))['aggregated_results'].keys())
    x = np.arange(len(feature_sets))
    w = 0.8 / len(model_names)

    for metric_name, mean_key, std_key in [
        ('MAE', 'mae_mean', 'mae_std'),
        ('RMSE', 'rmse_mean', 'rmse_std'),
        ('Euclidean Error', 'euclidean_mean', 'euclidean_std'),
    ]:
        fig, ax = plt.subplots(figsize=(13, 6))
        for i, model_name in enumerate(model_names):
            agg = all_models[model_name]['aggregated_results']
            means = [agg[fs][mean_key] for fs in feature_sets]
            stds = [agg[fs][std_key] for fs in feature_sets]
            offset = (i - len(model_names) / 2) * w + w / 2
            ax.bar(x + offset, means, w, yerr=stds, capsize=4,
                   label=model_name, color=MODEL_PALETTE.get(model_name, f'C{i}'), alpha=0.85)
        ax.set_xlabel('Feature Set')
        ax.set_ylabel(f'{metric_name} (m)')
        ax.set_title(f'{metric_name} — All Models (Mean ± Std)')
        ax.set_xticks(x)
        ax.set_xticklabels([get_feature_label(fs) for fs in feature_sets])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        safe = metric_name.lower().replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f'all_models_{safe}_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ All-model bar chart ({metric_name}) → {save_dir}/all_models_{safe}_bar.png")


def _plot_multimodel_cdf(all_models: dict, save_dir: str):
    """CDF per feature set for all models."""
    os.makedirs(save_dir, exist_ok=True)
    feature_sets = list(next(iter(all_models.values()))['all_errors'][0].keys())
    for fs in feature_sets:
        plt.figure(figsize=(9, 6))
        for model_name, data in all_models.items():
            errors = np.concatenate([run[fs] for run in data['all_errors']])
            sorted_err = np.sort(errors)
            cdf = np.arange(len(sorted_err)) / float(len(sorted_err))
            plt.plot(sorted_err, cdf, label=model_name, linewidth=2, color=MODEL_PALETTE.get(model_name))
        plt.xlabel('Positioning Error (m)')
        plt.ylabel('CDF')
        plt.title(f'CDF — {get_feature_label(fs)}')
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_models_cdf_{fs}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ All-model CDF ({fs}) → {save_dir}/all_models_cdf_{fs}.png")


def _plot_multimodel_boxplot(all_models: dict, save_dir: str):
    """Box plots per feature set for all models."""
    os.makedirs(save_dir, exist_ok=True)
    feature_sets = list(next(iter(all_models.values()))['all_errors'][0].keys())
    for fs in feature_sets:
        rows = []
        for model_name, data in all_models.items():
            errors = np.concatenate([run[fs] for run in data['all_errors']])
            for e in errors:
                rows.append({'Model': model_name, 'Error': e})
        df = pd.DataFrame(rows)
        plt.figure(figsize=(11, 6))
        palette = {m: MODEL_PALETTE.get(m, f'C{i}') for i, m in enumerate(all_models.keys())}
        sns.boxplot(x='Model', y='Error', data=df, palette=palette)
        plt.xlabel('Model')
        plt.ylabel('Positioning Error (m)')
        plt.title(f'Error Box-Plot — {get_feature_label(fs)}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_models_boxplot_{fs}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ All-model box-plot ({fs}) → {save_dir}/all_models_boxplot_{fs}.png")


def _plot_multimodel_timing(all_models: dict, save_dir: str):
    """Timing comparison for all models."""
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(all_models.keys())
    feature_sets = list(next(iter(all_models.values()))['all_timings'][0].keys())
    x = np.arange(len(feature_sets))
    w = 0.7 / len(model_names)
    n = len(model_names)

    for time_key, title in [('training_time_sec', 'Training Time'), ('prediction_time_sec', 'Prediction Time')]:
        fig, ax = plt.subplots(figsize=(13, 6))
        for i, model_name in enumerate(model_names):
            avg = _avg_timings(all_models[model_name])
            vals = [avg.get(fs, {}).get(time_key, 0) for fs in feature_sets]
            offset = (i - n / 2) * w + w / 2
            ax.bar(x + offset, vals, w, label=model_name, color=MODEL_PALETTE.get(model_name, f'C{i}'), alpha=0.85)
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'{title} — All Models')
        ax.set_xticks(x)
        ax.set_xticklabels([get_feature_label(fs) for fs in feature_sets])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        safe = title.lower().replace(' ', '_')
        plt.savefig(os.path.join(save_dir, f'all_models_{safe}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ All-model {title} chart → {save_dir}/all_models_{safe}.png")


def _plot_multimodel_training_history(all_models: dict, save_dir: str):
    """Training history comparison for all models."""
    os.makedirs(save_dir, exist_ok=True)
    feature_sets = list(next(iter(all_models.values()))['all_histories'][0].keys())
    for fs in feature_sets:
        plt.figure(figsize=(11, 6))
        for model_name, data in all_models.items():
            losses = [h[fs].history['val_loss'] for h in data['all_histories']]
            loss_arr = pad_and_stack(losses)
            mean_loss = np.nanmean(loss_arr, axis=0)
            std_loss = np.nanstd(loss_arr, axis=0)
            epochs = np.arange(len(mean_loss))
            color = MODEL_PALETTE.get(model_name)
            plt.plot(epochs, mean_loss, label=model_name, color=color, linewidth=2)
            plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.15, color=color)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss')
        plt.title(f'Validation Loss — {get_feature_label(fs)} — All Models')
        plt.legend()
        plt.gca().xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'all_models_val_loss_{fs}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ All-model val-loss history ({fs}) → {save_dir}/all_models_val_loss_{fs}.png")


def _plot_multimodel_params_vs_mae(all_models: dict, save_dir: str):
    """Parameter count vs MAE scatter plot."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (model_name, data) in enumerate(all_models.items()):
        agg = data['aggregated_results']
        color = MODEL_PALETTE.get(model_name, f'C{i}')
        first = True
        for fs, stats in agg.items():
            n_p = stats.get('n_params_mean') or stats.get('n_params')
            mae = stats.get('mae_mean')
            if n_p is not None and mae is not None:
                ax.scatter(n_p, mae, color=color, s=100, zorder=3, alpha=0.7,
                           label=model_name if first else "")
                ax.annotate(f'{get_feature_label(fs)}', (n_p, mae), fontsize=8, alpha=0.7)
                first = False
    ax.set_xlabel('Trainable Parameters')
    ax.set_ylabel('Mean MAE (m)')
    ax.set_title('Complexity vs Accuracy Efficiency')
    ax.grid(linestyle='--', alpha=0.35)
    ax.legend(title="Models", loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_params_vs_mae.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Params vs MAE scatter → {save_dir}/all_models_params_vs_mae.png")


def _save_all_models_csv(all_models: dict, save_dir: str):
    """Save aggregated results for all models to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for model_name, data in all_models.items():
        for fs, stats in data['aggregated_results'].items():
            rows.append({
                'model': model_name,
                'feature_set': get_feature_label(fs),
                'MAE_Mean': stats.get('mae_mean', np.nan),
                'MAE_Std': stats.get('mae_std', np.nan),
                'RMSE_Mean': stats.get('rmse_mean', np.nan),
                'RMSE_Std': stats.get('rmse_std', np.nan),
                'Euclidean_Mean': stats.get('euclidean_mean', np.nan),
                'Euclidean_Std': stats.get('euclidean_std', np.nan),
                'Training_Time_Mean_s': stats.get('training_time_mean', np.nan),
                'Training_Time_Std_s': stats.get('training_time_std', np.nan),
                'N_Runs': stats.get('n_runs', np.nan),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, 'all_models_aggregated.csv'), index=False)
    print(f"💾 All-model aggregated CSV → {save_dir}/all_models_aggregated.csv")

def _plot_multimodel_error_bars(all_models: dict, save_dir: str):
    """Generate error bar plots comparing models across feature sets."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect errors for all models
    all_models_errors = {}
    for model_name, data in all_models.items():
        all_models_errors[model_name] = collect_errors_over_runs(data['all_errors'])
    
    if not all_models_errors:
        return
    
    # Get feature sets
    feature_sets = list(all_models_errors.values())[0].keys()
    models = list(all_models_errors.keys())
    
    # Prepare data
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_sets))
    width = 0.8 / len(models)
    
    for idx, model_name in enumerate(models):
        means = []
        stds = []
        for fs in feature_sets:
            errors = all_models_errors[model_name][fs]
            means.append(np.mean(errors))
            stds.append(np.std(errors))
        
        offset = (idx - len(models)/2) * width
        ax.bar(x + offset, means, width, label=model_name, 
               yerr=stds, capsize=3, alpha=0.7, 
               color=MODEL_PALETTE.get(model_name))
    
    ax.set_xlabel('Feature Set')
    ax.set_ylabel('Mean Euclidean Error (m)')
    ax.set_title('Error Comparison Across Models (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels([get_feature_label(fs) for fs in feature_sets], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'error_bars_comparison.png'), 
                bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"   ✅ Saved error bar comparison → {save_dir}/error_bars_comparison.png")
    
def _plot_pooled_cdf(all_models: dict, save_dir: str):
    """Generate pooled CDF (all errors combined across feature sets)."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for model_name, data in all_models.items():
        all_errors = []
        for run_errors in data['all_errors']:
            for errors in run_errors.values():
                all_errors.extend(errors)
        
        errors = np.array(all_errors)
        sorted_err = np.sort(errors)
        cdf = np.arange(len(sorted_err)) / float(len(sorted_err))
        plt.plot(sorted_err, cdf, label=model_name, linewidth=2, 
                 color=MODEL_PALETTE.get(model_name))
    
    plt.xlabel('Positioning Error (meters)')
    plt.ylabel('CDF')
    plt.title('CDF Comparison (All Feature Sets Combined)')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pooled_cdf_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved pooled CDF → {save_dir}/pooled_cdf_comparison.png")

# ============================================================================
# run_multirun_comparison - Backward compatible function
# ============================================================================
def run_multirun_comparison(qnn_data, ann_data, comparison_dir, extra_models=None):
    """
    DEPRECATED: Use run_all_models_comparison instead.
    Keeping for backward compatibility only.
    """
    print("⚠️  Warning: run_multirun_comparison is deprecated.")
    print("   Use run_all_models_comparison() for all comparison plots.")
    
    all_models = {'Inloq': qnn_data, 'Classical ANN': ann_data}
    if extra_models:
        all_models.update(extra_models)
    
    # Redirect to unified function
    run_all_models_comparison(all_models, comparison_dir)

# Add to src/plots.py (append these functions)

def plot_grouped_bar(df_long, save_dir):
    """Grouped bar chart: all models × input dimension."""
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = df_long[~df_long['Model'].str.contains('quantum only')]
    fig, ax = plt.subplots(figsize=(13, 6))
    model_order = ['InLoq (Proposed)', 'Classical ANN', 'LSTM', 'GRU', 'BiLSTM']
    palette = [MODEL_PALETTE[m] for m in model_order]
    
    sns.barplot(data=df, x='Input Dimension', y='Parameters',
                hue='Model', hue_order=model_order,
                palette=palette, ax=ax)
    
    for container in ax.containers:
        ax.bar_label(container, padding=3, fmt='%d', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Input Dimension ($d$)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Total Trainable Parameters', fontsize=15, fontweight='bold')
    ax.set_ylim(0, df['Parameters'].max() * 1.20)
    ax.legend(title='Architecture', fontsize=12, title_fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(axis='x', labelsize=12, rotation=0)
    ax.tick_params(axis='y', labelsize=12)

    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'all_models_param_grouped_bar.pdf')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✅ Grouped bar chart → {path}')


def plot_combined_InLoq_analysis(input_dims, save_dir):
    """
    Combined Dual-Axis Plot: 
    - Stacked Bar (Left Y): Classical vs Quantum Parameter Counts
    - Line Plot (Right Y): Quantum Parameter % Fraction
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    from src.parameter_counter import params_inloq_breakdown, params_quantum_only
    
    os.makedirs(save_dir, exist_ok=True)
    
    classical_vals, quantum_vals, totals, fractions = [], [], [], []
    for d in input_dims:
        # Use the breakdown dictionary
        r = params_inloq_breakdown[d]
        classical = r['classical_pre'] + r['post_q1'] + r['post_q2'] + r['output']
        quantum = r['quantum']
        total = classical + quantum
        
        classical_vals.append(classical)
        quantum_vals.append(quantum)
        totals.append(total)
        fractions.append(100 * quantum / total)

    x = np.arange(len(input_dims))
    width = 0.5
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: stacked bars
    bars_c = ax1.bar(x, classical_vals, width, label='Classical Layers',
                     color='#1f77b4', alpha=0.8)
    bars_q = ax1.bar(x, quantum_vals, width, bottom=classical_vals,
                     label='Quantum Layer', color='#aec7e8', alpha=0.9, edgecolor='#1f77b4')

    # Add labels inside bars
    ax1.bar_label(bars_c, label_type='center', fontsize=14, fontweight='bold', color='white')
    ax1.bar_label(bars_q, label_type='center', fontsize=14, fontweight='bold')

    # Add total count on top
    for xi, tot in zip(x, totals):
        ax1.text(xi, tot + 2, str(tot), ha='center', va='bottom', 
                 fontsize=14, fontweight='bold', color='#1f77b4')

    ax1.set_xlabel('Input Dimension ($d$)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Trainable Parameters', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'd={d}' for d in input_dims])
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Secondary axis: percentage line
    ax2 = ax1.twinx()
    line = ax2.plot(x, fractions, color='purple', marker='D', markersize=8, 
                    linewidth=2.5, label='Quantum Fraction (%)', zorder=5)
    
    # Add percentage labels
    for xi, frac in zip(x, fractions):
        ax2.annotate(f'{frac:.1f}%', (xi, frac), xytext=(0, 10), 
                     textcoords='offset points', ha='center', 
                     fontsize=14, color='purple', fontweight='bold')

    ax2.set_ylabel('Quantum Fraction of Total (%)', fontsize=14, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_ylim(0, max(fractions) * 1.5)

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=15)

    ax1.tick_params(axis='x', labelsize=12, rotation=0)
    ax1.tick_params(axis='y', labelsize=12)

    for label in ax1.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontweight('bold')
        
    plt.tight_layout()
    path = os.path.join(save_dir, 'InLoq_combined_analysis.pdf')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'✅ Combined InLoq analysis plot → {path}')
    
# ============================================================================
# run_all_models_comparison - Master function for 5-model comparison
# ============================================================================

def run_all_models_comparison(all_model_data: dict, comparison_dir: str):
    """
    Generate the complete model comparison suite.
    ALL comparison plots go into comparison_dir/all_models/
    """
    save_dir = os.path.join(comparison_dir, 'all_models')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"📊 Generating ALL-MODEL comparison plots  ({len(all_model_data)} models)")
    print(f"   Models : {list(all_model_data.keys())}")
    print(f"   Output : {save_dir}")
    print(f"{'='*70}\n")

    # 1. Grouped bar charts (MAE / RMSE / Euclidean)
    print("  📊 Grouped bar charts (MAE / RMSE / Euclidean)...")
    _plot_multimodel_bar(all_model_data, save_dir)

    # 2. CDF per feature set
    print("  📊 CDF per feature set...")
    _plot_multimodel_cdf(all_model_data, save_dir)

    # 3. Box-plots per feature set
    print("  📊 Box-plots per feature set...")
    _plot_multimodel_boxplot(all_model_data, save_dir)

    # 4. Error bar comparison (mean ± std per feature set)
    print("  📊 Error bar comparison across models...")
    _plot_multimodel_error_bars(all_model_data, save_dir)

    # 5. Training & prediction time charts
    print("  📊 Training & prediction time charts...")
    _plot_multimodel_timing(all_model_data, save_dir)

    # 6. Validation-loss history per feature set
    print("  📈 Validation-loss history per feature set...")
    _plot_multimodel_training_history(all_model_data, save_dir)

    # 7. Parameter count vs MAE scatter
    print("  📊 Parameter count vs MAE scatter...")
    _plot_multimodel_params_vs_mae(all_model_data, save_dir)

    # 8. Pooled CDF (all errors combined)
    print("  📊 Pooled CDF (all errors combined)...")
    _plot_pooled_cdf(all_model_data, save_dir)

    # 9. Save aggregated CSV
    print("  💾 Saving aggregated CSV...")
    _save_all_models_csv(all_model_data, save_dir)

    print(f"\n✅ All-model comparison complete!  Results in {save_dir}")
      
    