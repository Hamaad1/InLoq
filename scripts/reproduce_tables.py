#!/usr/bin/env python
"""
Reproduce LaTeX tables from saved experiment results.
"""

import os
import sys
import pandas as pd

from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tables import save_latex_table, save_latex_table_best_marked, load_saved_results


def main():
    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_BASE = PROJECT_ROOT / "results_InLoq"
    PAPER_TABLES_DIR = PROJECT_ROOT / "paper" / "Tables"
    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # RESULTS_BASE = "results_InLoq"
    # PAPER_TABLES_DIR = "paper/Tables"
    # os.makedirs(PAPER_TABLES_DIR, exist_ok=True)
    
    print("="*70)
    print("Reproducing LaTeX Tables from Saved Results")
    print("="*70)
    
    # Load results
    print("\nLoading saved results...")
    all_model_data = load_saved_results(RESULTS_BASE)
    
    if not all_model_data:
        print("❌ No results found! Please run run_experiments.py first.")
        return
    
    print(f"\n✅ Loaded {len(all_model_data)} models: {list(all_model_data.keys())}")
    
    # Generate main comparison tables
    print("\n" + "="*70)
    print("Generating main comparison tables...")
    print("="*70)
    
    save_latex_table(all_model_data, PAPER_TABLES_DIR)
    save_latex_table_best_marked(all_model_data, PAPER_TABLES_DIR)
    
    # Generate per-model detailed tables
    print("\n" + "="*70)
    print("Generating per-model detailed tables...")
    print("="*70)
    
    for model_name, model_data in all_model_data.items():
        model_table_dir = os.path.join(PAPER_TABLES_DIR, model_name.replace(' ', '_'))
        os.makedirs(model_table_dir, exist_ok=True)
        
        rows = []
        for feat_name, metrics in model_data['aggregated_results'].items():
            rows.append({
                'Feature Set': feat_name,
                'MAE (m)': f"{metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}",
                'RMSE (m)': f"{metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}",
                'Euclidean (m)': f"{metrics['euclidean_mean']:.4f} ± {metrics['euclidean_std']:.4f}",
                'Train Time (s)': f"{metrics['training_time_mean']:.1f} ± {metrics['training_time_std']:.1f}",
                'N Runs': metrics['n_runs'],
            })
        
        df = pd.DataFrame(rows)
        latex_path = os.path.join(model_table_dir, f'{model_name}_detailed.tex')
        df.to_latex(latex_path, index=False, escape=False)
        print(f"  ✅ Saved: {latex_path}")
    
    # Final summary
    print("\n" + "="*70)
    print(f"✅ ALL TABLES SAVED TO: {PAPER_TABLES_DIR}")
    print("="*70)
    print("\n📊 Files generated:")
    print(f"  • {os.path.join(PAPER_TABLES_DIR, 'comparison_table.tex')}")
    print(f"  • {os.path.join(PAPER_TABLES_DIR, 'comparison_table_best_bold.tex')}")
    for model_name in all_model_data.keys():
        print(f"  • {os.path.join(PAPER_TABLES_DIR, model_name.replace(' ', '_'), f'{model_name}_detailed.tex')}")


if __name__ == '__main__':
    main()