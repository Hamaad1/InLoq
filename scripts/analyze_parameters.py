#!/usr/bin/env python
"""
Parameter analysis script.
Generates parameter comparison tables and plots for all models.
This is separate from training/experiments.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parameter_counter import build_summary, INPUT_DIMS
from src.plots import plot_grouped_bar, plot_combined_InLoq_analysis


def main():
    # Create output directory
    output_dir = Path(__file__).parent.parent / "paper" / "Plots" / "Parameter_Analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Parameter Count Analysis")
    print("="*70)
    
    # Build summary dataframes
    print("\nCalculating parameter counts...")
    df_wide, df_long = build_summary(INPUT_DIMS)
    
    # Display summary
    print("\n" + "="*70)
    print("Parameter Summary (Wide Format):")
    print("="*70)
    print(df_wide.to_string())
    
    print("\n" + "="*70)
    print("Parameter Summary (Long Format):")
    print("="*70)
    print(df_long.to_string())
    
    # Save CSV files
    csv_dir = Path(__file__).parent.parent / "paper" / "Tables" / "Parameter_Analysis"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    df_wide.to_csv(csv_dir / "parameter_summary_wide.csv", index=False)
    df_long.to_csv(csv_dir / "parameter_summary_long.csv", index=False)
    print(f"\n✅ CSV files saved to {csv_dir}")
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating parameter plots...")
    print("="*70)
    
    plot_grouped_bar(df_long, output_dir)
    plot_combined_InLoq_analysis(INPUT_DIMS, output_dir)
    
    print("\n" + "="*70)
    print(f"✅ All parameter analysis plots saved to {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()