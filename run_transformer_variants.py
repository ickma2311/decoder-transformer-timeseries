#!/usr/bin/env python3
"""
Evaluate all transformer variants separately on retail sales data.
"""

import numpy as np
import pandas as pd
import os
from models.transformer_models import (
    SimpleLSTMForecaster, 
    TransformerForecaster,
    LargeTransformerForecaster, 
    DecoderOnlyForecaster
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_transformer_variants():
    """Evaluate all neural model variants on retail sales data."""
    
    print("="*70)
    print("TRANSFORMER VARIANTS EVALUATION ON REAL-WORLD DATA")
    print("="*70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load retail sales data
    retail_data = np.load('data/retail_sales_values.npz')
    retail_metadata = pd.read_csv('data/retail_sales_processed.csv')
    
    train_values = retail_data['train_values']
    test_values = retail_data['test_values']
    
    results = []
    
    # Model configurations (fast parameters)
    models_config = {
        'LSTM': {
            'class': SimpleLSTMForecaster,
            'params': {
                'seq_len': 30,
                'pred_len': 10,
                'hidden_size': 32,
                'epochs': 15,
                'lr': 0.01
            }
        },
        'Transformer': {
            'class': TransformerForecaster,
            'params': {
                'seq_len': 20,
                'pred_len': 10,
                'd_model': 32,
                'nhead': 4,
                'num_layers': 1,
                'epochs': 10,
                'lr': 0.001
            }
        },
        'LargeTransformer': {
            'class': LargeTransformerForecaster,
            'params': {
                'seq_len': 24,
                'pred_len': 10,
                'd_model': 64,
                'nhead': 8,
                'num_layers': 2,
                'epochs': 12,
                'lr': 0.001
            }
        },
        'DecoderOnly': {
            'class': DecoderOnlyForecaster,
            'params': {
                'seq_len': 20,
                'd_model': 48,
                'nhead': 6,
                'num_layers': 2,
                'epochs': 10,
                'lr': 0.001
            }
        }
    }
    
    # Evaluate first 5 retail series
    print(f"\nEvaluating retail sales data (5 series)...")
    print("Models: LSTM, Transformer, LargeTransformer, DecoderOnly")
    print("-" * 70)
    
    for i in range(5):
        row = retail_metadata.iloc[i]
        series_id = row['series_id'] if 'series_id' in retail_metadata.columns else f'series_{i}'
        # Slice to actual lengths if available to avoid using padding
        train_len = int(row['train_length']) if 'train_length' in retail_metadata.columns else len(train_values[i])
        test_len = int(row['test_length']) if 'test_length' in retail_metadata.columns else len(test_values[i])
        train_data = train_values[i][:train_len]
        test_data = test_values[i][:test_len]
        
        print(f"\nSeries {i+1}/5: {series_id}")
        print(f"  Train length: {len(train_data)}, Test length: {len(test_data)}")
        
        series_result = {
            'series_id': series_id,
            'dataset': 'retail_sales',
            'train_length': len(train_data),
            'test_length': len(test_data)
        }
        
        # Evaluate each model
        for model_name, config in models_config.items():
            print(f"    {model_name}...", end=' ')
            try:
                # Adjust parameters for data length
                params = config['params'].copy()
                params['seq_len'] = min(params['seq_len'], len(train_data)//4)
                if 'pred_len' in params:  # Not all models have pred_len
                    params['pred_len'] = min(params['pred_len'], len(test_data))
                
                # Create and train model
                model = config['class'](**params)
                
                if model.fit(train_data):
                    predictions = model.predict(len(test_data))
                    if not np.any(np.isnan(predictions)) and len(predictions) == len(test_data):
                        mae = mean_absolute_error(test_data, predictions)
                        rmse = np.sqrt(mean_squared_error(test_data, predictions))
                        series_result[f'{model_name}_mae'] = mae
                        series_result[f'{model_name}_rmse'] = rmse
                        print(f"MAE: {mae:.3f}")
                    else:
                        series_result[f'{model_name}_mae'] = np.nan
                        series_result[f'{model_name}_rmse'] = np.nan
                        print("Invalid predictions")
                else:
                    series_result[f'{model_name}_mae'] = np.nan
                    series_result[f'{model_name}_rmse'] = np.nan
                    print("Fitting failed")
                    
            except Exception as e:
                series_result[f'{model_name}_mae'] = np.nan
                series_result[f'{model_name}_rmse'] = np.nan
                print(f"Error: {str(e)[:30]}...")
        
        results.append(series_result)
    
    # Save results
    df = pd.DataFrame(results)
    output_file = 'results/transformer_variants_results.csv'
    df.to_csv(output_file, index=False)
    
    # Print detailed summary
    print("\n" + "="*70)
    print("TRANSFORMER VARIANTS RESULTS SUMMARY")
    print("="*70)
    
    models = ['LSTM', 'Transformer', 'LargeTransformer', 'DecoderOnly']
    
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
                print(f"{model:15} | Mean: {mean_mae:6.3f} ± {std_mae:5.3f} | Range: [{min_mae:6.3f}, {max_mae:6.3f}] | Valid: {valid_count}/5")
            else:
                print(f"{model:15} | No valid results")
    
    print("\n" + "="*70)
    print("COMPARISON WITH TRADITIONAL MODELS")
    print("="*70)
    print("Traditional models on same retail sales data:")
    print("  ARIMA:   53.4 MAE (best traditional)")
    print("  Prophet: 76.9 MAE")
    print("  XGBoost: 58.9 MAE")
    print("  Linear:  69.3 MAE")
    
    # Detailed comparison analysis
    print("\n" + "="*70)
    print("NEURAL vs TRADITIONAL DETAILED ANALYSIS")
    print("="*70)
    
    arima_mae = 53.4
    for model in models:
        mae_col = f'{model}_mae'
        if mae_col in df.columns:
            valid_results = df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                diff = mean_mae - arima_mae
                pct_diff = (diff / arima_mae) * 100
                status = "✓ Better" if diff < 0 else "✗ Worse"
                print(f"{model:15} vs ARIMA: {diff:+6.1f} MAE ({pct_diff:+5.1f}%) {status}")
    
    # Transformer variants comparison
    print("\n" + "="*70)
    print("TRANSFORMER VARIANTS COMPARISON")
    print("="*70)
    
    transformer_models = ['Transformer', 'LargeTransformer', 'DecoderOnly']
    base_transformer_mae = None
    
    for model in transformer_models:
        mae_col = f'{model}_mae'
        if mae_col in df.columns:
            valid_results = df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                if model == 'Transformer':
                    base_transformer_mae = mean_mae
                    print(f"{model:15} | {mean_mae:6.3f} MAE (base)")
                else:
                    if base_transformer_mae is not None:
                        diff = mean_mae - base_transformer_mae
                        pct_diff = (diff / base_transformer_mae) * 100
                        status = "✓ Better" if diff < 0 else "✗ Worse"
                        print(f"{model:15} | {mean_mae:6.3f} MAE ({diff:+5.3f} vs base, {pct_diff:+4.1f}%) {status}")
                    else:
                        print(f"{model:15} | {mean_mae:6.3f} MAE")
    
    print(f"\nDetailed results saved to: {output_file}")
    return df

if __name__ == "__main__":
    evaluate_transformer_variants()
