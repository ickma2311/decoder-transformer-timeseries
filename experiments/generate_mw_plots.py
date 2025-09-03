#!/usr/bin/env python3
"""
Generate moving-window evaluation plots from comprehensive_moving_window_results.csv.

Outputs the standard figure filenames used by the paper so LaTeX/Markdown
referencing remains unchanged by default:
- results/figure1_overall_performance.png
- results/figure3_dataset_performance.png
- results/figure4_model_type_comparison.png

You can filter and/or write alternate files via CLI flags, e.g.:
- Only real-world datasets:  --dataset-type real_world --suffix _realworld
- Only synthetic datasets:   --dataset-type synthetic  --suffix _synthetic
- Specific datasets:         --include-datasets "Retail" "Energy"
- Specific models:           --include-models DecoderOnly LargeTransformer
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

RESULTS_CSV = 'results/comprehensive_moving_window_results.csv'

MODEL_TYPES = {
    'ARIMA': 'Traditional',
    'Prophet': 'Traditional',
    'Linear': 'Traditional',
    'XGBoost': 'Traditional',
    'LSTM': 'Neural',
    'Transformer': 'Neural',
    'LargeTransformer': 'Neural',
    'DecoderOnly': 'Neural',
}

def melt_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        base = {
            'series_id': r.get('series_id'),
            'dataset_name': r.get('dataset_name', r.get('dataset')),
            'dataset_type': r.get('dataset_type', np.nan),
            'window_size': r.get('window_size'),
            'n_windows': r.get('n_windows'),
            'evaluation_type': r.get('evaluation_type', 'unknown'),
        }
        for col in df.columns:
            if col.endswith('_mae'):
                model = col[:-4]
                val = r[col]
                if pd.notna(val):
                    rows.append({**base, 'model': model, 'mae': float(val),
                                 'model_type': MODEL_TYPES.get(model, 'Unknown')})
    return pd.DataFrame(rows)

def figure_overall_performance(long_df: pd.DataFrame, out_path: str):
    plt.style.use('default')
    plt.figure(figsize=(10, 6))
    order = long_df.groupby('model')['mae'].mean().sort_values().index
    sns.barplot(data=long_df, x='model', y='mae', order=order, ci='sd', capsize=0.1)
    plt.xlabel('Model')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Overall Performance (Moving-Window Evaluation)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure_dataset_performance(long_df: pd.DataFrame, out_path: str):
    plt.style.use('default')
    plt.figure(figsize=(12, 6))
    # Compute mean per dataset per model
    pivot = long_df.groupby(['dataset_name', 'model'])['mae'].mean().reset_index()
    # Keep consistent model order
    model_order = pivot.groupby('model')['mae'].mean().sort_values().index if not pivot.empty else None
    sns.barplot(data=pivot, x='dataset_name', y='mae', hue='model', hue_order=model_order)
    plt.xlabel('Dataset')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Dataset-Specific Performance (Moving-Window)')
    plt.xticks(rotation=20)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure_model_type(long_df: pd.DataFrame, out_path: str):
    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    type_stats = long_df.groupby('model_type')['mae'].agg(['mean', 'std']).reset_index()
    sns.barplot(data=type_stats, x='model_type', y='mean', ci=None)
    plt.xlabel('Model Type')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Model Type Comparison (Moving-Window)')
    for idx, row in type_stats.iterrows():
        plt.text(idx, row['mean'] + 0.01 * (type_stats['mean'].max() if len(type_stats) else 1.0), f"{row['mean']:.3f}",
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate moving-window plots with optional filters')
    parser.add_argument('--input-csv', default=RESULTS_CSV, help='Path to comprehensive_moving_window_results.csv')
    parser.add_argument('--output-dir', default='results', help='Directory for output figures')
    parser.add_argument('--suffix', default='', help='Filename suffix to append before .png (e.g., _realworld)')
    parser.add_argument('--dataset-type', choices=['synthetic', 'real_world'], help='Filter by dataset_type')
    parser.add_argument('--include-datasets', nargs='*', help='Filter to dataset names containing any of these substrings (case-insensitive)')
    parser.add_argument('--include-models', nargs='*', help='Only include these models (e.g., LSTM Transformer DecoderOnly)')
    parser.add_argument('--exclude-models', nargs='*', help='Exclude these models')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Missing {args.input_csv}. Run run_moving_window_comprehensive.py first.")
    df = pd.read_csv(args.input_csv)
    long_df = melt_results(df)
    if long_df.empty:
        raise ValueError('No model MAE columns found in results CSV.')

    # Apply filters
    if args.dataset_type:
        long_df = long_df[long_df['dataset_type'] == args.dataset_type]
    if args.include_datasets:
        needles = [s.lower() for s in args.include_datasets]
        long_df = long_df[long_df['dataset_name'].str.lower().apply(lambda s: any(n in s for n in needles))]
    if args.include_models:
        long_df = long_df[long_df['model'].isin(set(args.include_models))]
    if args.exclude_models:
        long_df = long_df[~long_df['model'].isin(set(args.exclude_models))]

    if long_df.empty:
        raise ValueError('No rows left after applying filters — check your flags.')

    os.makedirs(args.output_dir, exist_ok=True)
    sfx = args.suffix or ''
    fig1 = os.path.join(args.output_dir, f'figure1_overall_performance{sfx}.png')
    fig3 = os.path.join(args.output_dir, f'figure3_dataset_performance{sfx}.png')
    fig4 = os.path.join(args.output_dir, f'figure4_model_type_comparison{sfx}.png')

    figure_overall_performance(long_df, fig1)
    figure_dataset_performance(long_df, fig3)
    figure_model_type(long_df, fig4)

    print('Saved figures:')
    print(f' - {fig1}')
    print(f' - {fig3}')
    print(f' - {fig4}')

if __name__ == '__main__':
    main()
