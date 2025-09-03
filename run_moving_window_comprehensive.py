#!/usr/bin/env python3
"""
Comprehensive moving window evaluation across all datasets.
"""

import os
import pandas as pd
from models.traditional_models_moving_window import MovingWindowEvaluator
from models.neural_models_moving_window import MovingWindowNeuralEvaluator

def run_comprehensive_evaluation():
    """Run moving window evaluation on all datasets."""
    
    print("="*80)
    print("COMPREHENSIVE MOVING WINDOW EVALUATION")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Define all datasets
    datasets = [
        # Synthetic datasets
        ('data/trend_seasonal_values.npz', 'Synthetic - Trend + Seasonal', 'synthetic'),
        ('data/multi_seasonal_values.npz', 'Synthetic - Multi-Seasonal', 'synthetic'),
        ('data/random_walk_values.npz', 'Synthetic - Random Walk', 'synthetic'),
        
        # Real-world datasets
        ('data/retail_sales_values.npz', 'Real-World - Retail Sales', 'real_world'),
        ('data/energy_consumption_values.npz', 'Real-World - Energy Consumption', 'real_world')
    ]
    
    all_results = []
    
    # Initialize evaluators
    trad_evaluator = MovingWindowEvaluator(window_size=100, max_windows=20)
    neural_evaluator = MovingWindowNeuralEvaluator(window_size=100, max_windows=10)
    
    for dataset_path, dataset_name, dataset_type in datasets:
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset not found: {dataset_path}")
            continue
            
        print(f"\n{'='*80}")
        print(f"EVALUATING: {dataset_name}")
        print(f"{'='*80}")
        
        # Traditional models
        print("\n🔶 TRADITIONAL MODELS:")
        try:
            max_series = 3 if 'energy' in dataset_path else 5
            trad_results = trad_evaluator.evaluate_dataset_moving_window(dataset_path, max_series=max_series)
            
            for result in trad_results:
                result['dataset_name'] = dataset_name
                result['dataset_type'] = dataset_type
                result['evaluation_type'] = 'traditional'
                all_results.append(result)
                
        except Exception as e:
            print(f"❌ Traditional evaluation failed: {e}")
        
        # Neural models (skip energy due to computational cost)
        if 'energy' not in dataset_path:
            print("\n🔷 NEURAL MODELS:")
            try:
                neural_results = neural_evaluator.evaluate_dataset_moving_window(dataset_path, max_series=3)
                
                for result in neural_results:
                    result['dataset_name'] = dataset_name
                    result['dataset_type'] = dataset_type
                    result['evaluation_type'] = 'neural'
                    all_results.append(result)
                    
            except Exception as e:
                print(f"❌ Neural evaluation failed: {e}")
        else:
            print("\n🔷 NEURAL MODELS: Skipped (computational cost)")
    
    # Save comprehensive results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv('results/comprehensive_moving_window_results.csv', index=False)
        print(f"\n✅ Comprehensive results saved to results/comprehensive_moving_window_results.csv")
        
        # Generate summary by dataset
        print("\n" + "="*80)
        print("RESULTS SUMMARY BY DATASET")
        print("="*80)
        
        # Group by dataset
        for dataset_name in df['dataset_name'].unique():
            dataset_df = df[df['dataset_name'] == dataset_name]
            
            print(f"\n📊 {dataset_name}")
            print("-" * 60)
            
            # Traditional models
            trad_df = dataset_df[dataset_df['evaluation_type'] == 'traditional']
            if not trad_df.empty:
                print("  Traditional Models:")
                for model in ['ARIMA', 'Prophet', 'Linear', 'XGBoost']:
                    mae_col = f'{model}_mae'
                    if mae_col in trad_df.columns:
                        valid_results = trad_df[mae_col].dropna()
                        if len(valid_results) > 0:
                            mean_mae = valid_results.mean()
                            std_mae = valid_results.std()
                            print(f"    {model:8}: {mean_mae:6.3f} ± {std_mae:5.3f} MAE")
            
            # Neural models
            neural_df = dataset_df[dataset_df['evaluation_type'] == 'neural']
            if not neural_df.empty:
                print("  Neural Models:")
                # Auto-detect available neural model columns
                neural_models = sorted({c[:-4] for c in neural_df.columns if c.endswith('_mae')})
                for model in neural_models:
                    mae_col = f'{model}_mae'
                    if mae_col in neural_df.columns:
                        valid_results = neural_df[mae_col].dropna()
                        if len(valid_results) > 0:
                            mean_mae = valid_results.mean()
                            std_mae = valid_results.std()
                            print(f"    {model:15}: {mean_mae:6.3f} ± {std_mae:5.3f} MAE")
        
        # Overall comparison by dataset type
        print("\n" + "="*80)
        print("COMPARISON: SYNTHETIC vs REAL-WORLD")
        print("="*80)
        
        synthetic_df = df[df['dataset_type'] == 'synthetic']
        real_world_df = df[df['dataset_type'] == 'real_world']
        
        print("\n📈 SYNTHETIC DATASETS AVERAGE:")
        if not synthetic_df.empty:
            all_models = sorted({c[:-4] for c in synthetic_df.columns if c.endswith('_mae')})
            for model in all_models:
                mae_col = f'{model}_mae'
                valid_results = synthetic_df[mae_col].dropna() if mae_col in synthetic_df.columns else []
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    print(f"  {model:15}: {mean_mae:6.3f} MAE")
        
        print("\n🏢 REAL-WORLD DATASETS AVERAGE:")
        if not real_world_df.empty:
            all_models = sorted({c[:-4] for c in real_world_df.columns if c.endswith('_mae')})
            for model in all_models:
                mae_col = f'{model}_mae'
                valid_results = real_world_df[mae_col].dropna() if mae_col in real_world_df.columns else []
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    print(f"  {model:15}: {mean_mae:6.3f} MAE")
        
    else:
        print("\n❌ No results obtained")

if __name__ == "__main__":
    run_comprehensive_evaluation()
