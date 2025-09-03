"""
Traditional time series forecasting models with moving window validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

# Traditional models
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

class MovingWindowARIMA:
    """ARIMA with moving window validation."""
    
    def __init__(self, max_p=2, max_d=2, max_q=2):
        self.max_p = max_p
        self.max_d = max_d  
        self.max_q = max_q
        
    def _find_best_order(self, series: np.ndarray) -> Tuple[int, int, int]:
        """Find best ARIMA order using AIC (simplified for speed)."""
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Reduced search space for moving window efficiency
        for p in range(min(3, self.max_p + 1)):
            for d in range(min(2, self.max_d + 1)):
                for q in range(min(3, self.max_q + 1)):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
                        
        return best_order
    
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead from window."""
        try:
            if len(window) < 10:  # Minimum data requirement
                return np.nan
                
            # Find best order for this window
            order = self._find_best_order(window)
            
            # Fit model
            model = ARIMA(window, order=order)
            fitted = model.fit()
            
            # Predict next step
            forecast = fitted.forecast(steps=1)
            return float(forecast[0])
            
        except Exception as e:
            return np.nan

class MovingWindowProphet:
    """Prophet with moving window validation."""
    
    def __init__(self):
        self.prophet = None
        
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead from window."""
        try:
            if len(window) < 10:
                return np.nan
            
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(window), freq='D'),
                'y': window
            })
            
            # Fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True if len(window) > 14 else False,
                yearly_seasonality=False,
                interval_width=0.8
            )
            model.fit(df)
            
            # Predict next step
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            
            return float(forecast['yhat'].iloc[-1])
            
        except Exception as e:
            return np.nan

class MovingWindowLinear:
    """Linear trend model with moving window validation."""
    
    def __init__(self):
        self.model = LinearRegression()
        
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead using linear trend."""
        try:
            if len(window) < 5:
                return np.nan
            
            # Create features (time indices)
            X = np.arange(len(window)).reshape(-1, 1)
            y = window
            
            # Fit linear model
            self.model.fit(X, y)
            
            # Predict next step
            next_X = np.array([[len(window)]])
            prediction = self.model.predict(next_X)
            
            return float(prediction[0])
            
        except Exception as e:
            return np.nan

class MovingWindowXGBoost:
    """XGBoost with moving window validation."""
    
    def __init__(self, lag_features=5):
        self.lag_features = lag_features
        
    def _create_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for XGBoost."""
        X, y = [], []
        
        for i in range(self.lag_features, len(series)):
            # Use last lag_features values as features
            X.append(series[i-self.lag_features:i])
            y.append(series[i])
            
        return np.array(X), np.array(y)
        
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead using XGBoost."""
        try:
            if len(window) < self.lag_features + 5:
                return np.nan
            
            # Create features
            X, y = self._create_features(window)
            
            if len(X) < 3:  # Need minimum samples
                return np.nan
            
            # Train XGBoost model
            model = lgb.LGBMRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X, y)
            
            # Predict next step using last lag_features values
            last_features = window[-self.lag_features:].reshape(1, -1)
            prediction = model.predict(last_features)
            
            return float(prediction[0])
            
        except Exception as e:
            return np.nan

class MovingWindowEvaluator:
    """Evaluates traditional models using moving window approach."""
    
    def __init__(self, window_size=100, max_windows=50):
        self.window_size = window_size
        self.max_windows = max_windows
        
        self.models = {
            'ARIMA': MovingWindowARIMA(),
            'Prophet': MovingWindowProphet(),
            'Linear': MovingWindowLinear(),
            'XGBoost': MovingWindowXGBoost()
        }
        
    def evaluate_series_moving_window(self, series: np.ndarray, series_id: str) -> Dict[str, Any]:
        """Evaluate all models on a single series using moving windows."""
        
        # Determine window parameters
        actual_window_size = min(self.window_size, len(series) // 3)
        if actual_window_size < 20:
            return None  # Series too short
            
        # Calculate number of windows
        max_possible_windows = len(series) - actual_window_size - 1
        n_windows = min(self.max_windows, max_possible_windows)
        
        if n_windows < 5:
            return None  # Too few windows for reliable evaluation
            
        print(f"    Evaluating {series_id}: {n_windows} windows of size {actual_window_size}")
        
        results = {
            'series_id': series_id,
            'window_size': actual_window_size,
            'n_windows': n_windows,
            'series_length': len(series)
        }
        
        # For each model, collect predictions across all windows
        for model_name, model in self.models.items():
            predictions = []
            actuals = []
            
            print(f"      {model_name}...", end=' ')
            
            for i in range(n_windows):
                # Define window and target
                window = series[i:i + actual_window_size]
                target = series[i + actual_window_size]
                
                # Get prediction
                pred = model.predict_one_step(window)
                
                if not np.isnan(pred):
                    predictions.append(pred)
                    actuals.append(target)
            
            # Calculate metrics
            if len(predictions) > 0:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                
                results[f'{model_name}_mae'] = mae
                results[f'{model_name}_rmse'] = rmse
                results[f'{model_name}_valid_predictions'] = len(predictions)
                
                print(f"MAE: {mae:.3f} ({len(predictions)}/{n_windows} valid)")
            else:
                results[f'{model_name}_mae'] = np.nan
                results[f'{model_name}_rmse'] = np.nan
                results[f'{model_name}_valid_predictions'] = 0
                print("No valid predictions")
        
        return results
        
    def evaluate_dataset_moving_window(self, dataset_path: str, max_series: int = 5) -> List[Dict[str, Any]]:
        """Evaluate dataset using moving window approach."""
        print(f"\nMoving Window Evaluation: {dataset_path}")
        print("-" * 50)
        
        # Load data
        metadata_file = dataset_path.replace('_values.npz', '_processed.csv')
        values_file = dataset_path
        
        metadata_df = pd.read_csv(metadata_file)
        values_data = np.load(values_file, allow_pickle=True)
        
        train_values = values_data['train_values']
        
        # Use only training data for moving window (more realistic)
        dataset_results = []
        n_series = min(max_series, len(metadata_df))
        
        for i in range(n_series):
            row = metadata_df.iloc[i]
            series_data = np.array(train_values[i])
            
            # Slice to true length if available to avoid removing legitimate zeros
            if 'train_length' in metadata_df.columns:
                true_len = int(row['train_length'])
                series_data = series_data[:true_len]
            else:
                # Fallback: trim only trailing zeros (assumed padding)
                nz = np.nonzero(series_data)[0]
                if len(nz) > 0:
                    series_data = series_data[:nz[-1] + 1]
            
            result = self.evaluate_series_moving_window(series_data, row['series_id'])
            if result:
                result['dataset'] = row['dataset']
                dataset_results.append(result)
        
        return dataset_results

def main():
    """Test moving window evaluation on real-world data."""
    print("="*60)
    print("TRADITIONAL MODELS - MOVING WINDOW VALIDATION")
    print("="*60)
    
    evaluator = MovingWindowEvaluator(window_size=200, max_windows=30)
    
    # Test on retail sales first
    dataset_path = 'data/retail_sales_values.npz'
    results = evaluator.evaluate_dataset_moving_window(dataset_path, max_series=3)
    
    if results:
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('results/moving_window_traditional_results.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("MOVING WINDOW RESULTS SUMMARY")
        print("="*60)
        
        for model in ['ARIMA', 'Prophet', 'Linear', 'XGBoost']:
            mae_col = f'{model}_mae'
            if mae_col in df.columns:
                valid_results = df[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    std_mae = valid_results.std()
                    print(f"{model:8} | Mean MAE: {mean_mae:.3f} ± {std_mae:.3f} | Valid: {len(valid_results)}/{len(df)}")
                else:
                    print(f"{model:8} | No valid results")
    
    else:
        print("No results obtained")

if __name__ == "__main__":
    main()
