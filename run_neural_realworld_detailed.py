#!/usr/bin/env python3
"""
Run all neural network models on real-world retail sales data with detailed transformer variants.
"""

import numpy as np
import pandas as pd
import os
from models.neural_models_moving_window import MovingWindowNeuralEvaluator

def evaluate_realworld_detailed():
    """Evaluate all neural models including separate transformer variants."""
    
    print("="*70)
    print("DETAILED NEURAL NETWORK EVALUATION ON REAL-WORLD DATA")
    print("="*70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    evaluator = MovingWindowNeuralEvaluator()
    
    # Evaluate retail sales dataset with first 5 series
    dataset_path = 'data/retail_sales_values.npz'
    max_series = 5
    
    print(f"\nEvaluating retail sales dataset (first {max_series} series)...")
    print("Models: Transformer, LargeTransformer, DecoderOnly, LSTM")
    print("-" * 70)
    
    try:
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}")
            return
            
        results = evaluator.evaluate_dataset_moving_window(dataset_path, max_series=max_series)
        
        if results:
            # Save detailed results manually
            output_file = 'results/neural_realworld_detailed_results.csv'
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_file, index=False)
            
            # Print summary
            df = pd.DataFrame(results)
            print("\n" + "="*70)
            print("DETAILED NEURAL NETWORK RESULTS SUMMARY")
            print("="*70)
            
            # Derive available models from DataFrame columns
            models = sorted({col[:-4] for col in df.columns if col.endswith('_mae')})
            
            for model in models:
                mae_col = f'{model}_mae'
                if mae_col in df.columns:
                    valid_results = df[mae_col].dropna()
                    if len(valid_results) > 0:
                        mean_mae = valid_results.mean()
                        std_mae = valid_results.std()
                        min_mae = valid_results.min()
                        max_mae = valid_results.max()
                        valid_count = len(valid_results)
                        print(f"{model:15} | Mean: {mean_mae:6.3f} ± {std_mae:5.3f} | Range: [{min_mae:6.3f}, {max_mae:6.3f}] | Valid: {valid_count}/{max_series}")
                    else:
                        print(f"{model:15} | No valid results")
                else:
                    print(f"{model:15} | Column not found")
            
            print("\n" + "="*70)
            print("COMPARISON WITH TRADITIONAL MODELS")
            print("="*70)
            print("Traditional models on same retail sales data: (baseline numbers to be supplied)")
            
            # Detailed comparison analysis
            print("\n" + "="*70)
            print("NEURAL vs TRADITIONAL DETAILED ANALYSIS")
            print("="*70)
            
            # If you have ARIMA baseline, you can enable a comparison here
            
            print(f"\nDetailed results saved to: {output_file}")
            
        else:
            print("No results obtained from evaluation.")
            
    except Exception as e:
        print(f"Failed to evaluate dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_realworld_detailed()
