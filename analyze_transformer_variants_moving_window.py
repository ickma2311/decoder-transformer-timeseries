#!/usr/bin/env python3
"""
Analyze comprehensive moving window results with separated transformer variants.
"""

import pandas as pd
import numpy as np

def analyze_transformer_variants():
    """Analyze moving window results with all transformer variants."""
    
    print("="*80)
    print("TRANSFORMER VARIANTS - MOVING WINDOW VALIDATION ANALYSIS")
    print("="*80)
    
    # Load results
    df = pd.read_csv('results/comprehensive_moving_window_results.csv')
    
    # Separate neural results (contains all transformer variants)
    neural_df = df[df['evaluation_type'] == 'neural'].copy()
    trad_df = df[df['evaluation_type'] == 'traditional'].copy()
    
    print(f"\nDatasets evaluated:")
    print(f"- Traditional models: {len(trad_df)} series")
    print(f"- Neural models: {len(neural_df)} series")
    
    # All transformer variants analysis
    print("\n" + "="*80)
    print("ALL TRANSFORMER VARIANTS - MOVING WINDOW VALIDATION")
    print("="*80)
    
    # Extract all neural model columns
    neural_models = sorted({col[:-4] for col in neural_df.columns if col.endswith('_mae')})
    print(f"Neural models detected: {neural_models}")
    
    # Overall performance by model type
    print(f"\n{'='*20} OVERALL PERFORMANCE {'='*20}")
    all_results = []
    
    # Traditional models
    trad_models = ['ARIMA', 'Prophet', 'Linear', 'XGBoost']
    for model in trad_models:
        mae_col = f'{model}_mae'
        if mae_col in trad_df.columns:
            valid_results = trad_df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                all_results.append((model, mean_mae, 'Traditional', len(valid_results)))
    
    # Neural models
    for model in neural_models:
        mae_col = f'{model}_mae'
        if mae_col in neural_df.columns:
            valid_results = neural_df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                all_results.append((model, mean_mae, 'Neural', len(valid_results)))
    
    # Sort by performance
    all_results.sort(key=lambda x: x[1])
    
    print("\nOverall Ranking (Lower MAE = Better):")
    for i, (model, mae, model_type, count) in enumerate(all_results, 1):
        print(f"{i:2}. {model:15} ({model_type:11}) | {mae:6.3f} MAE | N={count}")
    
    # Transformer variants detailed comparison
    print("\n" + "="*80)
    print("TRANSFORMER VARIANTS DETAILED ANALYSIS")
    print("="*80)
    
    transformer_variants = [m for m in neural_models if 'transformer' in m.lower() or m == 'Transformer']
    print(f"Transformer variants: {transformer_variants}")
    
    # By dataset type
    for dataset_type in ['synthetic', 'real_world']:
        subset = neural_df[neural_df['dataset_type'] == dataset_type]
        if len(subset) == 0:
            continue
            
        print(f"\n{'='*15} {dataset_type.upper()} DATA {'='*15}")
        
        variant_results = []
        for model in neural_models:
            mae_col = f'{model}_mae'
            if mae_col in subset.columns:
                valid_results = subset[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    std_mae = valid_results.std()
                    count = len(valid_results)
                    variant_results.append((model, mean_mae, std_mae, count))
        
        # Sort by performance
        variant_results.sort(key=lambda x: x[1])
        
        for model, mean_mae, std_mae, count in variant_results:
            print(f"{model:15} | Mean: {mean_mae:6.3f} ± {std_mae:5.3f} MAE | N: {count}")
    
    # Dataset-specific performance
    print("\n" + "="*80)
    print("TRANSFORMER VARIANTS BY DATASET")
    print("="*80)
    
    for dataset_name in neural_df['dataset'].unique():
        if pd.isna(dataset_name):
            continue
            
        dataset_subset = neural_df[neural_df['dataset'] == dataset_name]
        if len(dataset_subset) == 0:
            continue
        
        print(f"\n{'='*10} {dataset_name.upper()} {'='*10}")
        
        dataset_results = []
        for model in neural_models:
            mae_col = f'{model}_mae'
            if mae_col in dataset_subset.columns:
                valid_results = dataset_subset[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    dataset_results.append((model, mean_mae))
        
        # Sort by performance
        dataset_results.sort(key=lambda x: x[1])
        
        for model, mean_mae in dataset_results:
            print(f"  {model:15}: {mean_mae:6.3f} MAE")
        
        # Compare with traditional models for this dataset
        dataset_trad = trad_df[trad_df['dataset'] == dataset_name]
        if len(dataset_trad) > 0:
            print("  Traditional comparison:")
            trad_results = []
            for model in trad_models:
                mae_col = f'{model}_mae'
                if mae_col in dataset_trad.columns:
                    mean_mae = dataset_trad[mae_col].mean()
                    if not np.isnan(mean_mae):
                        trad_results.append((model, mean_mae))
            
            trad_results.sort(key=lambda x: x[1])
            for model, mean_mae in trad_results[:3]:  # Show top 3
                print(f"    {model:13}: {mean_mae:6.3f} MAE")
    
    # Architecture comparison
    print("\n" + "="*80)
    print("TRANSFORMER ARCHITECTURE INSIGHTS")
    print("="*80)
    
    # Compare transformer variants
    base_transformer = None
    for model in neural_models:
        if model == 'Transformer':
            mae_col = f'{model}_mae'
            base_transformer = neural_df[mae_col].mean()
            break
    
    if base_transformer is not None and not np.isnan(base_transformer):
        print(f"\nBase Transformer: {base_transformer:.3f} MAE")
        
        for model in neural_models:
            if model != 'Transformer' and 'transformer' in model.lower():
                mae_col = f'{model}_mae'
                model_mae = neural_df[mae_col].mean()
                if not np.isnan(model_mae):
                    diff = model_mae - base_transformer
                    pct_diff = (diff / base_transformer) * 100
                    status = "✓ Better" if diff < 0 else "✗ Worse"
                    print(f"{model:15} vs Base: {diff:+6.3f} MAE ({pct_diff:+5.1f}%) {status}")
    
    # Best model per task
    print("\n" + "="*80)
    print("BEST NEURAL MODEL PER TASK")
    print("="*80)
    
    datasets = ['trend_seasonal', 'multi_seasonal', 'random_walk', 'retail_sales']
    for dataset_name in datasets:
        dataset_subset = neural_df[neural_df['dataset'] == dataset_name]
        if len(dataset_subset) == 0:
            continue
        
        best_model = None
        best_mae = float('inf')
        
        for model in neural_models:
            mae_col = f'{model}_mae'
            if mae_col in dataset_subset.columns:
                mean_mae = dataset_subset[mae_col].mean()
                if not np.isnan(mean_mae) and mean_mae < best_mae:
                    best_mae = mean_mae
                    best_model = model
        
        if best_model:
            print(f"{dataset_name:15}: {best_model:15} ({best_mae:.3f} MAE)")

if __name__ == "__main__":
    analyze_transformer_variants()