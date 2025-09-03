#!/usr/bin/env python3
"""
Run neural network models on real-world datasets.
"""

import numpy as np
import pandas as pd
import os
from models.transformer_models import TransformerModelEvaluator

def evaluate_realworld_neural():
    """Evaluate neural models on real-world datasets."""
    
    print("="*60)
    print("NEURAL NETWORKS ON REAL-WORLD DATA")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    evaluator = TransformerModelEvaluator()
    all_results = []
    
    # Real-world datasets  
    datasets = [
        ('data/retail_sales_values.npz', 'Retail Sales'),
        ('data/energy_consumption_values.npz', 'Energy Consumption')
    ]
    
    for dataset_path, dataset_name in datasets:
        print(f"\n{'='*20} {dataset_name} {'='*20}")
        
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"Dataset not found: {dataset_path}")
                continue
                
            # For retail sales: evaluate first 5 series (manageable size)
            # For energy: evaluate first 3 series (very large series)
            max_series = 5 if 'retail' in dataset_path else 3
            
            print(f"Evaluating first {max_series} series...")
            results = evaluator.evaluate_dataset(dataset_path, max_series=max_series)
            all_results.extend(results)
            
        except Exception as e:
            print(f"Failed to evaluate {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save and summarize results
    if all_results:
        output_file = 'results/neural_realworld_results.csv'
        evaluator.save_results(all_results, output_file)
        
        # Print summary
        df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("NEURAL NETWORK RESULTS SUMMARY")
        print("="*60)
        
        for model in ['Transformer', 'LSTM']:
            mae_col = f'{model}_mae'
            if mae_col in df.columns:
                valid_results = df[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    std_mae = valid_results.std()
                    valid_count = len(valid_results)
                    print(f"{model:12} | Mean MAE: {mean_mae:.3f} ± {std_mae:.3f} | Valid: {valid_count}/{len(df)}")
                else:
                    print(f"{model:12} | No valid results")
        
        print("\n" + "="*60)
        print("COMPARISON WITH TRADITIONAL MODELS")
        print("="*60)
        print("Traditional models on retail sales:")
        print("  ARIMA:   53.4 MAE")
        print("  Prophet: 76.9 MAE")
        print("  XGBoost: 58.9 MAE")
        print("  Linear:  69.3 MAE")
        
    else:
        print("\nNo results obtained.")

if __name__ == "__main__":
    evaluate_realworld_neural()