"""
Neural network models with moving window validation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

class TimeSeriesWindowDataset(Dataset):
    """PyTorch dataset for moving window time series."""
    
    def __init__(self, data: np.ndarray, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len] if idx + self.seq_len < len(self.data) else self.data[-1]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class MovingWindowLSTM:
    """LSTM with moving window validation."""
    
    def __init__(self, seq_len=20, hidden_size=32, epochs=10, lr=0.01):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def _create_model(self):
        """Create LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                out, _ = self.lstm(x.unsqueeze(-1))  # Add feature dimension
                out = self.fc(out[:, -1, :])  # Use last timestep
                return out
        
        return LSTMModel(1, self.hidden_size)
    
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead from window."""
        try:
            # Check minimum requirements
            if len(window) < self.seq_len + 5:
                return np.nan
            
            # Normalize data
            window_scaled = self.scaler.fit_transform(window.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesWindowDataset(window_scaled, self.seq_len)
            if len(dataset) == 0:
                return np.nan
                
            dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
            
            # Create and train model
            model = self._create_model().to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Generate prediction
            model.eval()
            with torch.no_grad():
                # Use last seq_len points for prediction
                last_sequence = torch.FloatTensor(window_scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
                prediction_scaled = model(last_sequence).cpu().item()
                
                # Denormalize prediction
                prediction = self.scaler.inverse_transform([[prediction_scaled]])[0, 0]
                
            return float(prediction)
            
        except Exception as e:
            return np.nan

class MovingWindowTransformer:
    """Simplified Transformer with moving window validation."""
    
    def __init__(self, seq_len=20, d_model=32, nhead=4, epochs=10, lr=0.001):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def _create_model(self):
        """Create simplified Transformer model."""
        class SimpleTransformerModel(nn.Module):
            def __init__(self, d_model, nhead, seq_len):
                super().__init__()
                self.d_model = d_model
                self.input_projection = nn.Linear(1, d_model)
                self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 2,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.output_projection = nn.Linear(d_model, 1)
                
            def forward(self, x):
                # x shape: [batch_size, seq_len]
                x = x.unsqueeze(-1)  # Add feature dim: [batch_size, seq_len, 1]
                x = self.input_projection(x)  # [batch_size, seq_len, d_model]
                
                # Add positional encoding
                x = x + self.pos_embedding.unsqueeze(0)
                
                # Apply transformer
                x = self.transformer(x)
                
                # Use last position for prediction
                x = self.output_projection(x[:, -1, :])
                
                return x
        
        return SimpleTransformerModel(self.d_model, self.nhead, self.seq_len)
    
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead from window."""
        try:
            # Check minimum requirements  
            if len(window) < self.seq_len + 5:
                return np.nan
            
            # Normalize data
            window_scaled = self.scaler.fit_transform(window.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesWindowDataset(window_scaled, self.seq_len)
            if len(dataset) == 0:
                return np.nan
                
            dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
            
            # Create and train model
            model = self._create_model().to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Generate prediction
            model.eval()
            with torch.no_grad():
                # Use last seq_len points for prediction
                last_sequence = torch.FloatTensor(window_scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
                prediction_scaled = model(last_sequence).cpu().item()
                
                # Denormalize prediction
                prediction = self.scaler.inverse_transform([[prediction_scaled]])[0, 0]
                
            return float(prediction)
            
        except Exception as e:
            return np.nan

class MovingWindowLargeTransformer:
    """Large Transformer with moving window validation."""
    
    def __init__(self, seq_len=20, d_model=128, nhead=8, num_layers=4, epochs=10, lr=0.001):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def _create_model(self):
        """Create large transformer model."""
        class LargeTransformerModel(nn.Module):
            def __init__(self, d_model, nhead, num_layers, seq_len):
                super().__init__()
                self.d_model = d_model
                self.input_projection = nn.Linear(1, d_model)
                self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,  # Larger FFN
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                # Deeper output projection
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 2, d_model // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 4, 1)
                )
                
            def forward(self, x):
                # x shape: [batch_size, seq_len]
                x = x.unsqueeze(-1)  # Add feature dim: [batch_size, seq_len, 1]
                x = self.input_projection(x)  # [batch_size, seq_len, d_model]
                
                # Add positional encoding
                x = x + self.pos_embedding.unsqueeze(0)
                
                # Apply transformer
                x = self.transformer(x)
                
                # Use last position for prediction
                x = self.output_projection(x[:, -1, :])
                
                return x
        
        return LargeTransformerModel(self.d_model, self.nhead, self.num_layers, self.seq_len)
    
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead from window."""
        try:
            # Check minimum requirements  
            if len(window) < self.seq_len + 5:
                return np.nan
            
            # Normalize data
            window_scaled = self.scaler.fit_transform(window.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesWindowDataset(window_scaled, self.seq_len)
            if len(dataset) == 0:
                return np.nan
                
            dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
            
            # Create and train model
            model = self._create_model().to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Generate prediction
            model.eval()
            with torch.no_grad():
                # Use last seq_len points for prediction
                last_sequence = torch.FloatTensor(window_scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
                prediction_scaled = model(last_sequence).cpu().item()
                
                # Denormalize prediction
                prediction = self.scaler.inverse_transform([[prediction_scaled]])[0, 0]
                
            return float(prediction)
            
        except Exception as e:
            return np.nan

class MovingWindowDecoderOnly:
    """Decoder-only Transformer with moving window validation."""
    
    def __init__(self, seq_len=20, d_model=64, nhead=4, num_layers=3, epochs=10, lr=0.001):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
    def _create_model(self):
        """Create causal masked self-attention decoder-only model (no cross-attention)."""
        class DecoderOnlyModel(nn.Module):
            def __init__(self, d_model, nhead, num_layers, seq_len):
                super().__init__()
                self.d_model = d_model
                self.seq_len = seq_len
                self.input_projection = nn.Linear(1, d_model)
                self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                # Output projection
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 2, 1)
                )
                
            def _generate_causal_mask(self, seq_len):
                mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
                return mask.to(next(self.parameters()).device)
                
            def forward(self, x):
                # x shape: [batch_size, seq_len]
                batch_size, seq_len = x.shape
                x = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
                x = self.input_projection(x)  # [batch_size, seq_len, d_model]
                
                # Add positional encoding
                x = x + self.pos_embedding.unsqueeze(0)
                
                # Causal self-attention mask
                causal_mask = self._generate_causal_mask(seq_len)
                
                # Apply masked self-attention stack
                x = self.transformer(x, mask=causal_mask)
                
                # Use last position for prediction
                x = self.output_projection(x[:, -1, :])
                
                return x
        
        return DecoderOnlyModel(self.d_model, self.nhead, self.num_layers, self.seq_len)
    
    def predict_one_step(self, window: np.ndarray) -> float:
        """Predict one step ahead using autoregressive approach."""
        try:
            # Check minimum requirements  
            if len(window) < self.seq_len + 5:
                return np.nan
            
            # Normalize data
            window_scaled = self.scaler.fit_transform(window.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesWindowDataset(window_scaled, self.seq_len)
            if len(dataset) == 0:
                return np.nan
                
            dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
            
            # Create and train model
            model = self._create_model().to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Generate prediction
            model.eval()
            with torch.no_grad():
                # Use last seq_len points for prediction
                last_sequence = torch.FloatTensor(window_scaled[-self.seq_len:]).unsqueeze(0).to(self.device)
                prediction_scaled = model(last_sequence).cpu().item()
                
                # Denormalize prediction
                prediction = self.scaler.inverse_transform([[prediction_scaled]])[0, 0]
                
            return float(prediction)
            
        except Exception as e:
            return np.nan

class MovingWindowNeuralEvaluator:
    """Evaluates neural models using moving window approach."""
    
    def __init__(self, window_size=100, max_windows=20):
        self.window_size = window_size
        self.max_windows = max_windows
        
        self.models = {
            'LSTM': MovingWindowLSTM(seq_len=20, epochs=8),
            'Transformer': MovingWindowTransformer(seq_len=20, epochs=8),
            'LargeTransformer': MovingWindowLargeTransformer(seq_len=20, epochs=8),
            'DecoderOnly': MovingWindowDecoderOnly(seq_len=20, epochs=8)
        }
        
    def evaluate_series_moving_window(self, series: np.ndarray, series_id: str) -> Dict[str, Any]:
        """Evaluate neural models on a single series using moving windows."""
        
        # Determine window parameters
        actual_window_size = min(self.window_size, len(series) // 3)
        if actual_window_size < 30:  # Neural networks need more data
            return None
            
        # Calculate number of windows (fewer for neural networks due to training time)
        max_possible_windows = len(series) - actual_window_size - 1
        n_windows = min(self.max_windows, max_possible_windows)
        
        if n_windows < 3:
            return None
            
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
        
    def evaluate_dataset_moving_window(self, dataset_path: str, max_series: int = 3) -> List[Dict[str, Any]]:
        """Evaluate dataset using moving window approach."""
        print(f"\nNeural Moving Window Evaluation: {dataset_path}")
        print("-" * 50)
        
        # Load data
        metadata_file = dataset_path.replace('_values.npz', '_processed.csv')
        values_file = dataset_path
        
        metadata_df = pd.read_csv(metadata_file)
        values_data = np.load(values_file, allow_pickle=True)
        
        train_values = values_data['train_values']
        
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
    """Test neural moving window evaluation on real-world data."""
    print("="*60)
    print("NEURAL MODELS - MOVING WINDOW VALIDATION")
    print("="*60)
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    evaluator = MovingWindowNeuralEvaluator(window_size=150, max_windows=10)
    
    # Test on retail sales first
    dataset_path = 'data/retail_sales_values.npz'
    results = evaluator.evaluate_dataset_moving_window(dataset_path, max_series=2)
    
    if results:
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('results/moving_window_neural_results.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("NEURAL MOVING WINDOW RESULTS SUMMARY")
        print("="*60)
        
        for model in ['LSTM', 'Transformer', 'LargeTransformer', 'DecoderOnly']:
            mae_col = f'{model}_mae'
            if mae_col in df.columns:
                valid_results = df[mae_col].dropna()
                if len(valid_results) > 0:
                    mean_mae = valid_results.mean()
                    std_mae = valid_results.std()
                    print(f"{model:15} | Mean MAE: {mean_mae:.3f} ± {std_mae:.3f} | Valid: {len(valid_results)}/{len(df)}")
                else:
                    print(f"{model:15} | No valid results")
                    
        # Compare with traditional results if available
        try:
            trad_df = pd.read_csv('results/moving_window_traditional_results.csv')
            print("\n" + "="*60)
            print("COMPARISON WITH TRADITIONAL MODELS")
            print("="*60)
            print("Traditional (Moving Window):")
            for model in ['ARIMA', 'Prophet', 'XGBoost']:
                mae_col = f'{model}_mae'
                if mae_col in trad_df.columns:
                    mean_mae = trad_df[mae_col].mean()
                    print(f"  {model:8}: {mean_mae:.3f} MAE")
            
            print("\\nNeural (Moving Window):")
            for model in ['LSTM', 'Transformer', 'LargeTransformer', 'DecoderOnly']:
                mae_col = f'{model}_mae'
                if mae_col in df.columns:
                    valid_results = df[mae_col].dropna()
                    if len(valid_results) > 0:
                        mean_mae = valid_results.mean()
                        print(f"  {model:15}: {mean_mae:.3f} MAE")
                        
        except:
            pass
    
    else:
        print("No results obtained")

if __name__ == "__main__":
    main()
