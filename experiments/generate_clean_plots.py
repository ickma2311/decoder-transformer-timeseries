#!/usr/bin/env python3
"""
Generate clean, publication-ready plots from moving-window results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set publication-quality style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

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

# Clean model names for display
MODEL_DISPLAY_NAMES = {
    'ARIMA': 'ARIMA',
    'Prophet': 'Prophet',
    'Linear': 'Linear',
    'XGBoost': 'XGBoost',
    'LSTM': 'LSTM', 
    'Transformer': 'Transformer',
    'LargeTransformer': 'Large Transformer',
    'DecoderOnly': 'Decoder-Only'
}

# Color palette - distinguishable colors
COLORS = {
    'Traditional': '#2E86AB',  # Blue
    'Neural': '#A23B72'        # Purple/Pink
}

def melt_results(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format results to long format for plotting."""
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
                    rows.append({
                        **base, 
                        'model': model, 
                        'mae': float(val),
                        'model_type': MODEL_TYPES.get(model, 'Unknown'),
                        'model_display': MODEL_DISPLAY_NAMES.get(model, model)
                    })
    return pd.DataFrame(rows)

def figure1_overall_performance(long_df: pd.DataFrame, out_path: str):
    """Generate overall performance ranking plot."""
    plt.figure(figsize=(12, 8))
    
    # Calculate means and sort
    model_stats = long_df.groupby(['model', 'model_display', 'model_type'])['mae'].agg(['mean', 'std']).reset_index()
    model_stats = model_stats.sort_values('mean')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(model_stats))
    colors = [COLORS[model_type] for model_type in model_stats['model_type']]
    
    bars = ax.bar(x_pos, model_stats['mean'], yerr=model_stats['std'], 
                  capsize=5, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, model_stats['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title('Overall Performance Rankings (Moving-Window Evaluation)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_stats['model_display'], rotation=45, ha='right')
    
    # Add ranking numbers
    for i, (x, rank) in enumerate(zip(x_pos, range(1, len(model_stats) + 1))):
        ax.text(x, -0.3, f'#{rank}', ha='center', va='top', fontweight='bold', fontsize=12)
    
    # Legend
    legend_elements = [Rectangle((0,0),1,1, facecolor=COLORS[mt], alpha=0.8, edgecolor='black') 
                      for mt in ['Traditional', 'Neural']]
    ax.legend(legend_elements, ['Traditional', 'Neural'], 
             loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure3_dataset_performance(long_df: pd.DataFrame, out_path: str):
    """Generate dataset-specific performance plot."""
    # Clean up dataset names
    long_df = long_df.copy()
    long_df['dataset_clean'] = long_df['dataset_name'].str.replace('Synthetic - ', '').str.replace('Real-World - ', '')
    
    plt.figure(figsize=(14, 8))
    
    # Compute mean per dataset per model
    pivot = long_df.groupby(['dataset_clean', 'model_display', 'model_type'])['mae'].mean().reset_index()
    
    # Order models by overall performance
    model_order = pivot.groupby('model_display')['mae'].mean().sort_values().index
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a more sophisticated color palette
    palette = {}
    traditional_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    neural_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    trad_models = [m for m in model_order if any(pivot[pivot['model_display']==m]['model_type'] == 'Traditional')]
    neural_models = [m for m in model_order if any(pivot[pivot['model_display']==m]['model_type'] == 'Neural')]
    
    for i, model in enumerate(trad_models):
        palette[model] = traditional_colors[i % len(traditional_colors)]
    for i, model in enumerate(neural_models):
        palette[model] = neural_colors[i % len(neural_colors)]
    
    sns.barplot(data=pivot, x='dataset_clean', y='mae', hue='model_display', 
               hue_order=model_order, palette=palette, ax=ax)
    
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title('Dataset-Specific Performance (Moving-Window Evaluation)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for readability
    ax.tick_params(axis='x', rotation=45)
    
    # Improve legend
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left',
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure4_model_type_comparison(long_df: pd.DataFrame, out_path: str):
    """Generate model type comparison plot."""
    plt.figure(figsize=(10, 6))
    
    # Calculate statistics by model type
    type_stats = long_df.groupby('model_type')['mae'].agg(['mean', 'std', 'count']).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    x_pos = np.arange(len(type_stats))
    colors = [COLORS[mt] for mt in type_stats['model_type']]
    
    bars = ax.bar(x_pos, type_stats['mean'], yerr=type_stats['std'],
                  capsize=8, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, mean_val, count) in enumerate(zip(bars, type_stats['mean'], type_stats['count'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'n={count}', ha='center', va='center', fontweight='bold', 
                color='white', fontsize=10)
    
    ax.set_xlabel('Model Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold') 
    ax.set_title('Traditional vs Neural Model Performance\n(Moving-Window Evaluation)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(type_stats['model_type'])
    
    # Add performance gap annotation
    if len(type_stats) == 2:
        trad_mae = type_stats[type_stats['model_type'] == 'Traditional']['mean'].iloc[0]
        neural_mae = type_stats[type_stats['model_type'] == 'Neural']['mean'].iloc[0]
        gap = abs(neural_mae - trad_mae) / max(trad_mae, neural_mae) * 100
        better = 'Traditional' if trad_mae < neural_mae else 'Neural'
        ax.text(0.5, max(type_stats['mean']) * 0.8, 
                f'{better} methods\nperform {gap:.1f}% better',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all plots with clean formatting."""
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Missing {RESULTS_CSV}. Run run_moving_window_comprehensive.py first.")
    
    df = pd.read_csv(RESULTS_CSV)
    long_df = melt_results(df)
    
    if long_df.empty:
        raise ValueError('No model MAE columns found in results CSV.')
    
    print(f"Found {len(long_df)} data points across {long_df['model'].nunique()} models")
    print("Model performance summary:")
    print(long_df.groupby('model_display')['mae'].agg(['mean', 'std', 'count']).round(3))
    
    os.makedirs('results', exist_ok=True)
    
    # Generate plots
    figure1_overall_performance(long_df, 'results/figure1_overall_performance.png')
    figure3_dataset_performance(long_df, 'results/figure3_dataset_performance.png') 
    figure4_model_type_comparison(long_df, 'results/figure4_model_type_comparison.png')
    
    print('\nGenerated clean plots:')
    print('- results/figure1_overall_performance.png')
    print('- results/figure3_dataset_performance.png')
    print('- results/figure4_model_type_comparison.png')

if __name__ == '__main__':
    main()