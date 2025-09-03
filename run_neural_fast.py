#!/usr/bin/env python3
"""
Run neural network models on real-world datasets with fast parameters.
"""

import numpy as np
import pandas as pd
import os
from models.transformer_models import SimpleLSTMForecaster, TransformerForecaster
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_realworld_fast():
    """Fast evaluation of neural models on real-world data."""
    
    print("="*60)
    print("FAST NEURAL NETWORK EVALUATION ON REAL-WORLD DATA")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load retail sales data
    retail_data = np.load('data/retail_sales_values.npz')
    retail_metadata = pd.read_csv('data/retail_sales_processed.csv')
    
    train_values = retail_data['train_values']
    test_values = retail_data['test_values']
    
    results = []
    
    # Evaluate first 5 retail series with fast parameters
    print("\nEvaluating retail sales data (5 series)...")
    for i in range(5):
        series_id = retail_metadata.iloc[i]['series_id']
        train_data = train_values[i]
        test_data = test_values[i]
        
        print(f"  Series {i+1}/5: {series_id}")
        
        series_result = {
            'series_id': series_id,
            'dataset': 'retail_sales',
            'train_length': len(train_data),
            'test_length': len(test_data)
        }
        
        # LSTM with fast parameters
        print(f"    LSTM...", end=' ')
        try:
            lstm = SimpleLSTMForecaster(
                seq_len=min(30, len(train_data)//4),
                pred_len=min(10, len(test_data)),
                hidden_size=32,  # Reduced
                epochs=15,       # Reduced
                lr=0.01         # Increased for faster convergence
            )
            
            if lstm.fit(train_data):
                predictions = lstm.predict(len(test_data))
                if not np.any(np.isnan(predictions)) and len(predictions) == len(test_data):
                    mae = mean_absolute_error(test_data, predictions)
                    rmse = np.sqrt(mean_squared_error(test_data, predictions))
                    series_result['LSTM_mae'] = mae
                    series_result['LSTM_rmse'] = rmse
                    print(f"MAE: {mae:.3f}")
                else:
                    series_result['LSTM_mae'] = np.nan
                    series_result['LSTM_rmse'] = np.nan
                    print("Invalid predictions")
            else:
                series_result['LSTM_mae'] = np.nan
                series_result['LSTM_rmse'] = np.nan
                print("Fitting failed")
        except Exception as e:
            series_result['LSTM_mae'] = np.nan
            series_result['LSTM_rmse'] = np.nan
            print(f"Error: {str(e)[:30]}...")
        
        # Simple Transformer with very fast parameters
        print(f"    Transformer...", end=' ')
        try:
            transformer = TransformerForecaster(
                seq_len=min(20, len(train_data)//4),
                pred_len=min(10, len(test_data)),
                d_model=32,      # Reduced
                nhead=4,         # Reduced
                num_layers=1,    # Reduced
                epochs=10,       # Reduced
                lr=0.001
            )
            
            if transformer.fit(train_data):
                predictions = transformer.predict(len(test_data))
                if not np.any(np.isnan(predictions)) and len(predictions) == len(test_data):
                    mae = mean_absolute_error(test_data, predictions)
                    rmse = np.sqrt(mean_squared_error(test_data, predictions))
                    series_result['Transformer_mae'] = mae
                    series_result['Transformer_rmse'] = rmse
                    print(f"MAE: {mae:.3f}")
                else:
                    series_result['Transformer_mae'] = np.nan
                    series_result['Transformer_rmse'] = np.nan
                    print("Invalid predictions")
            else:
                series_result['Transformer_mae'] = np.nan
                series_result['Transformer_rmse'] = np.nan
                print("Fitting failed")
        except Exception as e:
            series_result['Transformer_mae'] = np.nan
            series_result['Transformer_rmse'] = np.nan
            print(f"Error: {str(e)[:30]}...")
        
        results.append(series_result)
    
    # Save results
    df = pd.DataFrame(results)
    output_file = 'results/neural_realworld_fast_results.csv'
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("NEURAL NETWORK RESULTS SUMMARY")
    print("="*60)
    
    for model in ['LSTM', 'Transformer']:
        mae_col = f'{model}_mae'
        if mae_col in df.columns:
            valid_results = df[mae_col].dropna()
            if len(valid_results) > 0:
                mean_mae = valid_results.mean()
                std_mae = valid_results.std()
                min_mae = valid_results.min()
                max_mae = valid_results.max()
                valid_count = len(valid_results)
                print(f"{model:12} | Mean: {mean_mae:.3f} ± {std_mae:.3f} | Range: [{min_mae:.3f}, {max_mae:.3f}] | Valid: {valid_count}/5")
            else:
                print(f"{model:12} | No valid results")
    
    print("\n" + "="*60)
    print("COMPARISON WITH TRADITIONAL MODELS")
    print("="*60)
    print("Traditional models on same retail sales data:")
    print("  ARIMA:   53.4 MAE (best traditional)")
    print("  Prophet: 76.9 MAE")
    print("  XGBoost: 58.9 MAE")
    print("  Linear:  69.3 MAE")
    
    # Performance analysis
    lstm_results = df['LSTM_mae'].dropna()
    transformer_results = df['Transformer_mae'].dropna()
    
    if len(lstm_results) > 0 and len(transformer_results) > 0:
        print("\n" + "="*60)
        print("NEURAL vs TRADITIONAL ANALYSIS")
        print("="*60)
        
        lstm_vs_arima = lstm_results.mean() - 53.4
        transformer_vs_arima = transformer_results.mean() - 53.4
        
        print(f"LSTM vs ARIMA (best traditional):      {lstm_vs_arima:+.1f} MAE")
        print(f"Transformer vs ARIMA (best traditional): {transformer_vs_arima:+.1f} MAE")
        
        if lstm_vs_arima > 0:
            print("→ LSTM performs worse than best traditional method")
        else:
            print("→ LSTM outperforms best traditional method!")
            
        if transformer_vs_arima > 0:
            print("→ Transformer performs worse than best traditional method")  
        else:
            print("→ Transformer outperforms best traditional method!")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    evaluate_realworld_fast()