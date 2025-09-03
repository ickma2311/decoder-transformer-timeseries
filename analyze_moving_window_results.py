#!/usr/bin/env python3
"""
Analyze comprehensive moving window validation results.
"""

import pandas as pd
import numpy as np

def analyze_moving_window_results():
    """Analyze moving window evaluation results."""
    
    print("="*80)
    print("COMPREHENSIVE MOVING WINDOW RESULTS ANALYSIS")
    print("="*80)
    
    # Load results
    df = pd.read_csv('results/comprehensive_moving_window_results.csv')
    
    # Separate traditional and neural results
    trad_df = df[df['evaluation_type'] == 'traditional'].copy()
    neural_df = df[df['evaluation_type'] == 'neural'].copy()
    
    print(f"\nDatasets evaluated:")
    print(f"- Traditional models: {len(trad_df)} series")
    print(f"- Neural models: {len(neural_df)} series")
    
    # Traditional models analysis
    print("\n" + "="*80)
    print("TRADITIONAL MODELS - MOVING WINDOW VALIDATION")
    print("="*80)
    
    trad_models = ['ARIMA', 'Prophet', 'Linear', 'XGBoost']
    
    for dataset_type in ['synthetic', 'real_world']:
        subset = trad_df[trad_df['dataset_type'] == dataset_type]
        if len(subset) == 0:
            continue
            
        print(f"\n{'='*20} {dataset_type.upper()} DATA {'='*20}")
        
        for model in trad_models:
            mae_col = f'{model}_mae'
            if mae_col in subset.columns:
                valid_results = subset[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    std_mae = valid_results.std()
                    min_mae = valid_results.min()
                    max_mae = valid_results.max()
                    count = len(valid_results)
                    print(f"{model:12} | Mean: {mean_mae:6.3f} ± {std_mae:5.3f} | Range: [{min_mae:6.3f}, {max_mae:6.3f}] | N: {count}")
                else:
                    print(f"{model:12} | No valid results")
    
    # Neural models analysis
    print("\n" + "="*80)
    print("NEURAL MODELS - MOVING WINDOW VALIDATION")
    print("="*80)
    
    neural_models = ['LSTM', 'Transformer']
    
    for dataset_type in ['synthetic', 'real_world']:
        subset = neural_df[neural_df['dataset_type'] == dataset_type]
        if len(subset) == 0:
            continue
            
        print(f"\n{'='*20} {dataset_type.upper()} DATA {'='*20}")
        
        for model in neural_models:
            mae_col = f'{model}_mae'
            if mae_col in subset.columns:
                valid_results = subset[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    std_mae = valid_results.std()
                    min_mae = valid_results.min()
                    max_mae = valid_results.max()
                    count = len(valid_results)
                    print(f"{model:12} | Mean: {mean_mae:6.3f} ± {std_mae:5.3f} | Range: [{min_mae:6.3f}, {max_mae:6.3f}] | N: {count}")
                else:
                    print(f"{model:12} | No valid results")
    
    # Dataset-specific analysis
    print("\n" + "="*80)
    print("DATASET-SPECIFIC PERFORMANCE")
    print("="*80)
    
    for dataset_name in df['dataset'].unique():
        if pd.isna(dataset_name):
            continue
            
        dataset_trad = trad_df[trad_df['dataset'] == dataset_name]
        dataset_neural = neural_df[neural_df['dataset'] == dataset_name]
        
        print(f"\n{'='*15} {dataset_name.upper()} {'='*15}")
        
        if len(dataset_trad) > 0:
            print("Traditional Models:")
            for model in trad_models:
                mae_col = f'{model}_mae'
                if mae_col in dataset_trad.columns:
                    mean_mae = dataset_trad[mae_col].mean()
                    if not np.isnan(mean_mae):
                        print(f"  {model:12}: {mean_mae:6.3f} MAE")
        
        if len(dataset_neural) > 0:
            print("Neural Models:")
            for model in neural_models:
                mae_col = f'{model}_mae'
                if mae_col in dataset_neural.columns:
                    mean_mae = dataset_neural[mae_col].mean()
                    if not np.isnan(mean_mae):
                        print(f"  {model:12}: {mean_mae:6.3f} MAE")
    
    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL MODEL COMPARISON (MOVING WINDOW)")
    print("="*80)
    
    all_models = []
    
    # Traditional models
    for model in trad_models:
        mae_col = f'{model}_mae'
        if mae_col in trad_df.columns:
            valid_results = trad_df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                all_models.append((model, mean_mae, 'Traditional'))
    
    # Neural models  
    for model in neural_models:
        mae_col = f'{model}_mae'
        if mae_col in neural_df.columns:
            valid_results = neural_df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                all_models.append((model, mean_mae, 'Neural'))
    
    # Sort by performance
    all_models.sort(key=lambda x: x[1])
    
    print("\nRanking (Lower MAE = Better):")
    for i, (model, mae, model_type) in enumerate(all_models, 1):
        print(f"{i:2}. {model:12} ({model_type:11}) | {mae:6.3f} MAE")
    
    # Real-world vs Synthetic comparison
    print("\n" + "="*80)
    print("REAL-WORLD vs SYNTHETIC DATA PERFORMANCE")
    print("="*80)
    
    print("\nTraditional Models:")
    for model in trad_models:
        mae_col = f'{model}_mae'
        
        synthetic_mae = trad_df[trad_df['dataset_type'] == 'synthetic'][mae_col].mean()
        real_world_mae = trad_df[trad_df['dataset_type'] == 'real_world'][mae_col].mean()
        
        if not np.isnan(synthetic_mae) and not np.isnan(real_world_mae):
            diff = real_world_mae - synthetic_mae
            pct_diff = (diff / synthetic_mae) * 100
            print(f"{model:12} | Synthetic: {synthetic_mae:6.3f} | Real-world: {real_world_mae:6.3f} | Diff: {diff:+6.3f} ({pct_diff:+5.1f}%)")
    
    print("\nNeural Models:")
    for model in neural_models:
        mae_col = f'{model}_mae'
        
        synthetic_mae = neural_df[neural_df['dataset_type'] == 'synthetic'][mae_col].mean()
        real_world_mae = neural_df[neural_df['dataset_type'] == 'real_world'][mae_col].mean()
        
        if not np.isnan(synthetic_mae) and not np.isnan(real_world_mae):
            diff = real_world_mae - synthetic_mae
            pct_diff = (diff / synthetic_mae) * 100
            print(f"{model:12} | Synthetic: {synthetic_mae:6.3f} | Real-world: {real_world_mae:6.3f} | Diff: {diff:+6.3f} ({pct_diff:+5.1f}%)")

if __name__ == "__main__":
    analyze_moving_window_results()