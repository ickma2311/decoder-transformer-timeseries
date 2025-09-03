#!/usr/bin/env python3
"""
Generate a focused plot highlighting the key finding: Transformer is nearly optimal.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_key_finding_plot():
    """Create a focused plot showing top-performing models."""
    
    # Manual data from your moving window results
    results = {
        'Model': ['ARIMA', 'Transformer', 'Decoder-Only', 'Large Transformer', 'LSTM'],
        'MAE': [3.947, 4.029, 4.196, 4.570, 4.729], 
        'Type': ['Traditional', 'Neural', 'Neural', 'Neural', 'Neural'],
        'Gap_vs_ARIMA': [0.0, 0.082, 0.249, 0.623, 0.782],
        'Gap_Percent': [0.0, 2.1, 6.3, 15.8, 19.8]
    }
    
    df = pd.DataFrame(results)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Absolute Performance
    colors = ['#2E86AB' if t == 'Traditional' else '#A23B72' for t in df['Type']]
    bars1 = ax1.bar(range(len(df)), df['MAE'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, mae, gap) in enumerate(zip(bars1, df['MAE'], df['Gap_Percent'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mae:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Add ranking
        ax1.text(i, -0.2, f'#{i+1}', ha='center', va='top', fontweight='bold', fontsize=14)
        
        # Add gap percentage for neural models
        if i > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'+{gap:.1f}%', ha='center', va='center', fontweight='bold', 
                    color='white', fontsize=10)
    
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax1.set_title('Top 5 Model Rankings\n(Moving-Window Evaluation)', fontsize=16, fontweight='bold')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight key insight
    ax1.annotate('Only 2.1% gap!', xy=(1, df.loc[1, 'MAE']), xytext=(1.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Plot 2: Performance Gap vs ARIMA
    bars2 = ax2.bar(range(1, len(df)), df['Gap_Percent'][1:], 
                   color=['#A23B72']*4, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars2, df['Gap_Percent'][1:])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{gap:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylabel('Performance Gap vs ARIMA (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Neural Model Performance Gaps\n(vs Best Traditional Method)', fontsize=16, fontweight='bold')
    ax2.set_xticks(range(1, len(df)))
    ax2.set_xticklabels(df['Model'][1:], rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add threshold line at 5%
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(2, 5.5, 'Competitive Threshold (~5%)', ha='center', fontsize=10, 
             color='red', fontweight='bold')
    
    # Highlight Transformer
    ax2.annotate('Nearly Optimal!', xy=(0, df.loc[1, 'Gap_Percent']), xytext=(0.5, 10),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Add legends
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', alpha=0.8, label='Traditional'),
                      Patch(facecolor='#A23B72', alpha=0.8, label='Neural')]
    ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('results/key_finding_transformer_nearly_optimal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated key finding plot: results/key_finding_transformer_nearly_optimal.png")

if __name__ == '__main__':
    create_key_finding_plot()