"""
Simplified traditional time series forecasting models: ARIMA and Prophet.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Traditional models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

class ARIMAForecaster:
    """ARIMA forecasting model with automatic order selection."""
    
    def __init__(self, max_p=2, max_d=1, max_q=2):
        self.max_p = max_p
        self.max_d = max_d  
        self.max_q = max_q
        self.model = None
        self.fitted_order = None
        
    def _find_best_order(self, series: np.ndarray) -> Tuple[int, int, int]:
        """Find best ARIMA order using AIC."""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
                        
        return best_order
    
    def fit(self, series: np.ndarray):
        """Fit ARIMA model to time series."""
        try:
            # Find best order
            self.fitted_order = self._find_best_order(series)
            
            # Fit model with best order
            self.model = ARIMA(series, order=self.fitted_order)
            self.fitted_model = self.model.fit()
            
            return True
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts."""
        if not hasattr(self, 'fitted_model') or self.fitted_model is None:
            return np.array([np.nan] * steps)
            
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return np.array(forecast)
        except:
            return np.array([np.nan] * steps)

class ProphetForecaster:
    """Facebook Prophet forecasting model."""
    
    def __init__(self, yearly_seasonality=False, weekly_seasonality=False, 
                 daily_seasonality=True, changepoint_prior_scale=0.05):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        
    def fit(self, series: np.ndarray, frequency='H'):
        """Fit Prophet model to time series."""
        try:
            # Create DataFrame in Prophet format
            dates = pd.date_range(start='2020-01-01', periods=len(series), freq=frequency)
            df = pd.DataFrame({
                'ds': dates,
                'y': series
            })
            
            # Initialize and fit model
            self.model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality, 
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale
            )
            
            self.model.fit(df)
            self.last_date = dates[-1]
            self.frequency = frequency
            
            return True
        except Exception as e:
            print(f"Prophet fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts."""
        if self.model is None:
            return np.array([np.nan] * steps)
            
        try:
            # Create future dataframe
            future_dates = pd.date_range(
                start=self.last_date + pd.Timedelta('1H'), 
                periods=steps, 
                freq=self.frequency
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate forecast
            forecast = self.model.predict(future_df)
            return forecast['yhat'].values
            
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            return np.array([np.nan] * steps)

class SimpleLinearForecaster:
    """Simple linear trend extrapolation as baseline."""
    
    def __init__(self, window=20):
        self.window = window
        self.slope = None
        self.intercept = None
        self.last_value = None
        self.last_index = None
        
    def fit(self, series: np.ndarray):
        """Fit linear trend to recent data."""
        try:
            # Use last 'window' points for trend estimation
            recent_data = series[-self.window:]
            x = np.arange(len(recent_data))
            
            # Simple linear regression
            self.slope = np.polyfit(x, recent_data, 1)[0]
            self.last_value = series[-1]
            self.last_index = len(series) - 1
            
            return True
        except Exception as e:
            print(f"Linear forecaster fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate linear trend forecasts."""
        if self.slope is None:
            return np.array([np.nan] * steps)
            
        try:
            predictions = []
            for i in range(1, steps + 1):
                pred = self.last_value + self.slope * i
                predictions.append(pred)
            
            return np.array(predictions)
        except:
            return np.array([np.nan] * steps)

class XGBoostForecaster:
    """XGBoost forecasting model for time series with feature engineering."""
    
    def __init__(self, window=10, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.window = window
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _create_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for XGBoost."""
        from sklearn.preprocessing import StandardScaler
        
        X, y = [], []
        
        for i in range(self.window, len(series)):
            # Lagged values as features
            features = []
            for lag in range(1, self.window + 1):
                features.append(series[i - lag])
            
            # Statistical features over the window
            window_data = series[i - self.window:i]
            features.extend([
                np.mean(window_data),
                np.std(window_data),
                np.min(window_data),
                np.max(window_data),
                window_data[-1] - window_data[0]  # trend
            ])
            
            X.append(features)
            y.append(series[i])
        
        X, y = np.array(X), np.array(y)
        
        # Scale features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        return X_scaled, y_scaled
    
    def fit(self, series: np.ndarray):
        """Fit XGBoost model to time series."""
        try:
            if len(series) <= self.window + 1:
                print(f"Series too short for XGBoost (need >{self.window} points)")
                return False
            
            # Create features and targets
            X, y = self._create_features(series)
            
            if len(X) == 0:
                return False
            
            # Initialize and fit XGBoost
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=0
            )
            
            self.model.fit(X, y)
            self.last_window = series[-self.window:]
            
            return True
            
        except Exception as e:
            print(f"XGBoost fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts using XGBoost."""
        if self.model is None or self.scaler_X is None or self.scaler_y is None:
            return np.array([np.nan] * steps)
        
        try:
            predictions = []
            current_window = self.last_window.copy()
            
            for _ in range(steps):
                # Create features for current window
                features = []
                for lag in range(1, self.window + 1):
                    features.append(current_window[-lag])
                
                # Statistical features
                features.extend([
                    np.mean(current_window),
                    np.std(current_window),
                    np.min(current_window),
                    np.max(current_window),
                    current_window[-1] - current_window[0]
                ])
                
                # Scale features
                features_scaled = self.scaler_X.transform([features])
                
                # Predict next value
                pred_scaled = self.model.predict(features_scaled)[0]
                pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
                
                predictions.append(pred)
                
                # Update window with prediction
                current_window = np.append(current_window[1:], pred)
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"XGBoost prediction failed: {e}")
            return np.array([np.nan] * steps)

class TraditionalModelEvaluator:
    """Evaluates traditional forecasting models on time series datasets."""
    
    def __init__(self):
        self.models = {
            'ARIMA': ARIMAForecaster,
            'Prophet': ProphetForecaster, 
            'Linear': SimpleLinearForecaster,
            'XGBoost': XGBoostForecaster
        }
        self.results = []
    
    def evaluate_single_series(self, train_data: np.ndarray, test_data: np.ndarray,
                              series_id: str, dataset_name: str) -> Dict[str, Any]:
        """Evaluate all models on a single time series."""
        
        series_results = {
            'series_id': series_id,
            'dataset': dataset_name,
            'train_length': len(train_data),
            'test_length': len(test_data)
        }
        
        for model_name, model_class in self.models.items():
            print(f"    {model_name}...", end=' ')
            
            try:
                # Initialize and fit model
                model = model_class()
                fit_success = model.fit(train_data)
                
                if not fit_success:
                    series_results[f'{model_name}_mae'] = np.nan
                    series_results[f'{model_name}_rmse'] = np.nan
                    series_results[f'{model_name}_predictions'] = [np.nan] * len(test_data)
                    print("FAILED")
                    continue
                
                # Generate predictions
                predictions = model.predict(len(test_data))
                
                # Calculate metrics
                if not np.any(np.isnan(predictions)):
                    mae = mean_absolute_error(test_data, predictions)
                    rmse = np.sqrt(mean_squared_error(test_data, predictions))
                    
                    series_results[f'{model_name}_mae'] = mae
                    series_results[f'{model_name}_rmse'] = rmse
                    series_results[f'{model_name}_predictions'] = predictions.tolist()
                    print(f"MAE: {mae:.3f}")
                else:
                    series_results[f'{model_name}_mae'] = np.nan
                    series_results[f'{model_name}_rmse'] = np.nan
                    series_results[f'{model_name}_predictions'] = [np.nan] * len(test_data)
                    print("NaN predictions")
                    
            except Exception as e:
                series_results[f'{model_name}_mae'] = np.nan
                series_results[f'{model_name}_rmse'] = np.nan
                series_results[f'{model_name}_predictions'] = [np.nan] * len(test_data)
                print(f"ERROR: {str(e)[:50]}...")
        
        return series_results
    
    def evaluate_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Evaluate models on entire dataset."""
        print(f"\nEvaluating traditional models on: {dataset_path}")
        
        # Load data
        metadata_file = dataset_path.replace('_values.npz', '_processed.csv')
        values_file = dataset_path
        
        metadata_df = pd.read_csv(metadata_file)
        values_data = np.load(values_file, allow_pickle=True)
        
        train_values = values_data['train_values']
        test_values = values_data['test_values']
        
        dataset_results = []
        
        for i, row in metadata_df.iterrows():
            print(f"  Series {i+1}/{len(metadata_df)}: {row['series_id']}")
            
            train_data = np.array(train_values[i])
            test_data = np.array(test_values[i])
            
            series_result = self.evaluate_single_series(
                train_data, test_data, row['series_id'], row['dataset']
            )
            
            dataset_results.append(series_result)
        
        return dataset_results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save evaluation results to CSV."""
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

def main():
    """Main evaluation pipeline for traditional models."""
    print("="*60)
    print("TRADITIONAL MODELS EVALUATION")
    print("="*60)
    
    evaluator = TraditionalModelEvaluator()
    all_results = []
    
    # Evaluate on all datasets
    datasets = [
        'data/tourism_values.npz',
        'data/traffic_values.npz',
        'data/electricity_values.npz',
        'data/weather_values.npz',
        'data/ett_h1_values.npz'
    ]
    
    for dataset_path in datasets:
        try:
            results = evaluator.evaluate_dataset(dataset_path)
            all_results.extend(results)
        except Exception as e:
            print(f"Failed to evaluate {dataset_path}: {e}")
    
    # Save combined results
    if all_results:
        evaluator.save_results(all_results, 'results/traditional_models_results.csv')
        
        # Print summary statistics
        df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model in ['ARIMA', 'Prophet', 'Linear', 'XGBoost']:
            mae_col = f'{model}_mae'
            if mae_col in df.columns:
                mean_mae = df[mae_col].mean()
                valid_count = df[mae_col].notna().sum()
                print(f"{model:12} | Mean MAE: {mean_mae:.3f} | Valid: {valid_count}/{len(df)}")
    
    else:
        print("No results to save.")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    main()
