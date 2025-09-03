"""
Transformer-based time series forecasting models.
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

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series forecasting."""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TimeSeriesTransformer(nn.Module):
    """Simple Transformer for time series forecasting."""
    
    def __init__(self, input_dim: int = 1, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, seq_len: int = 50, pred_len: int = 10, 
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_len)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Add feature dimension
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Use last timestep for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Output projection
        out = self.output_projection(x)  # (batch_size, pred_len)
        
        return out

class LargeTimeSeriesTransformer(nn.Module):
    """Larger Transformer for time series forecasting with more parameters."""
    
    def __init__(self, input_dim: int = 1, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, seq_len: int = 50, pred_len: int = 10, 
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input projection with larger embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Larger transformer encoder with more layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,  # Larger FFN
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Deeper output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, pred_len)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Add feature dimension
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Use last timestep for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Output projection
        out = self.output_projection(x)  # (batch_size, pred_len)
        
        return out

class TransformerForecaster:
    """Wrapper for Transformer-based forecasting."""
    
    def __init__(self, seq_len: int = 50, pred_len: int = 10, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, epochs: int = 50, lr: float = 0.001):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.epochs = epochs
        self.lr = lr
        
        # Model parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Model and training components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, series: np.ndarray):
        """Fit transformer model to time series."""
        try:
            # Check minimum length requirement
            if len(series) < self.seq_len + self.pred_len + 10:
                print(f"Series too short: {len(series)} < {self.seq_len + self.pred_len + 10}")
                return False
            
            # Normalize data
            series_scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesDataset(series_scaled, self.seq_len, self.pred_len)
            
            if len(dataset) == 0:
                print("Empty dataset after processing")
                return False
            
            dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            # Initialize model
            self.model = TimeSeriesTransformer(
                input_dim=1,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
                pred_len=self.pred_len
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                # Print progress occasionally
                if epoch % 10 == 0:
                    avg_loss = total_loss / batch_count if batch_count > 0 else 0
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Store last sequence for prediction
            self.last_sequence = series_scaled[-self.seq_len:]
            
            return True
            
        except Exception as e:
            print(f"Transformer fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts using the trained model."""
        if self.model is None:
            return np.array([np.nan] * steps)
        
        try:
            self.model.eval()
            predictions = []
            current_seq = self.last_sequence.copy()
            
            with torch.no_grad():
                # Predict in chunks of pred_len
                remaining_steps = steps
                
                while remaining_steps > 0:
                    # Prepare input
                    input_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    pred = self.model(input_tensor).cpu().numpy().flatten()
                    
                    # Take only what we need
                    steps_to_take = min(remaining_steps, len(pred))
                    predictions.extend(pred[:steps_to_take])
                    
                    # Update sequence for next prediction
                    if remaining_steps > len(pred):
                        # Slide window forward with predictions
                        current_seq = np.concatenate([current_seq[len(pred):], pred])
                    
                    remaining_steps -= steps_to_take
            
            # Denormalize predictions
            predictions = np.array(predictions[:steps])
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions
            
        except Exception as e:
            print(f"Transformer prediction failed: {e}")
            return np.array([np.nan] * steps)

class LargeTransformerForecaster:
    """Wrapper for Large Transformer-based forecasting."""
    
    def __init__(self, seq_len: int = 50, pred_len: int = 10, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 6, epochs: int = 50, lr: float = 0.0005):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.epochs = epochs
        self.lr = lr
        
        # Model parameters (larger)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Model and training components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, series: np.ndarray):
        """Fit large transformer model to time series."""
        try:
            # Check minimum length requirement
            if len(series) < self.seq_len + self.pred_len + 10:
                print(f"Series too short: {len(series)} < {self.seq_len + self.pred_len + 10}")
                return False
            
            # Normalize data
            series_scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesDataset(series_scaled, self.seq_len, self.pred_len)
            
            if len(dataset) == 0:
                print("Empty dataset after processing")
                return False
            
            # Use smaller batch size for larger model
            batch_size = min(16, len(dataset))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Initialize large model
            self.model = LargeTimeSeriesTransformer(
                input_dim=1,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                seq_len=self.seq_len,
                pred_len=self.pred_len
            ).to(self.device)
            
            # Training setup with lower learning rate
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                scheduler.step()
                
                # Print progress occasionally
                if epoch % 10 == 0:
                    avg_loss = total_loss / batch_count if batch_count > 0 else 0
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Store last sequence for prediction
            self.last_sequence = series_scaled[-self.seq_len:]
            
            return True
            
        except Exception as e:
            print(f"Large Transformer fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts using the trained large model."""
        if self.model is None:
            return np.array([np.nan] * steps)
        
        try:
            self.model.eval()
            predictions = []
            current_seq = self.last_sequence.copy()
            
            with torch.no_grad():
                # Predict in chunks of pred_len
                remaining_steps = steps
                
                while remaining_steps > 0:
                    # Prepare input
                    input_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    pred = self.model(input_tensor).cpu().numpy().flatten()
                    
                    # Take only what we need
                    steps_to_take = min(remaining_steps, len(pred))
                    predictions.extend(pred[:steps_to_take])
                    
                    # Update sequence for next prediction
                    if remaining_steps > len(pred):
                        # Slide window forward with predictions
                        current_seq = np.concatenate([current_seq[len(pred):], pred])
                    
                    remaining_steps -= steps_to_take
            
            # Denormalize predictions
            predictions = np.array(predictions[:steps])
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions
            
        except Exception as e:
            print(f"Large Transformer prediction failed: {e}")
            return np.array([np.nan] * steps)

class DecoderOnlyTransformer(nn.Module):
    """Causal masked self-attention transformer for autoregressive forecasting.

    Implements a decoder-only style stack using masked self-attention only
    (no cross-attention/memory), preventing any future information leakage.
    """
    
    def __init__(self, input_dim: int = 1, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, max_seq_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Masked self-attention stack (encoder used as decoder-only with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection (predicts next value at each position)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def _generate_causal_mask(self, seq_len: int):
        """Generate an upper-triangular causal mask (True blocks attention)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        return mask
    
    def forward(self, x, return_attention: bool = False):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Add feature dimension
        x = x.unsqueeze(-1)  # (batch_size, seq_len, input_dim=1)
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Causal self-attention mask (S, S)
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)
        
        # Masked self-attention stack
        x = self.transformer(x, mask=causal_mask)  # (batch_size, seq_len, d_model)
        
        # Predict next value for each position
        output = self.output_projection(x)  # (batch_size, seq_len, 1)
        
        return output.squeeze(-1)  # (batch_size, seq_len)

class DecoderOnlyForecaster:
    """Wrapper for Decoder-only Transformer with autoregressive forecasting."""
    
    def __init__(self, seq_len: int = 50, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2, epochs: int = 50, lr: float = 0.001):
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        
        # Model parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Model and training components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, series: np.ndarray):
        """Fit decoder-only transformer with teacher forcing."""
        try:
            if len(series) < self.seq_len + 10:
                print(f"Series too short: {len(series)} < {self.seq_len + 10}")
                return False
            
            # Normalize data
            series_scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
            
            # Create sequences for teacher forcing
            X, y = [], []
            for i in range(self.seq_len, len(series_scaled)):
                X.append(series_scaled[i-self.seq_len:i])
                y.append(series_scaled[i])
            
            if len(X) == 0:
                print("No training sequences created")
                return False
            
            X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
            
            # Create dataset and dataloader
            dataset = list(zip(X, y))
            dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            # Initialize model
            self.model = DecoderOnlyTransformer(
                input_dim=1,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                max_seq_len=self.seq_len * 2
            ).to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
            # Training loop with teacher forcing
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                batch_count = 0
                
                for batch_x, batch_y in dataloader:
                    batch_x = torch.FloatTensor(batch_x).to(self.device)
                    batch_y = torch.FloatTensor(batch_y).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass: predict next value for each position
                    output = self.model(batch_x)
                    
                    # Use only the last prediction (next value after sequence)
                    pred = output[:, -1]
                    loss = criterion(pred, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                # Print progress
                if epoch % 10 == 0:
                    avg_loss = total_loss / batch_count if batch_count > 0 else 0
                    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            # Store last sequence for autoregressive prediction
            self.last_sequence = series_scaled[-self.seq_len:]
            
            return True
            
        except Exception as e:
            print(f"Decoder-only fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts using autoregressive prediction."""
        if self.model is None:
            return np.array([np.nan] * steps)
        
        try:
            self.model.eval()
            predictions = []
            
            # Start with the last sequence
            current_sequence = self.last_sequence.copy()
            
            with torch.no_grad():
                for _ in range(steps):
                    # Prepare input
                    input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                    
                    # Get prediction for next time step
                    output = self.model(input_tensor)
                    next_pred = output[0, -1].item()  # Last position prediction
                    
                    predictions.append(next_pred)
                    
                    # Update sequence: remove first element, add prediction
                    current_sequence = np.append(current_sequence[1:], next_pred)
            
            # Denormalize predictions
            predictions = np.array(predictions)
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions
            
        except Exception as e:
            print(f"Decoder-only prediction failed: {e}")
            return np.array([np.nan] * steps)

class SimpleLSTMForecaster:
    """Simple LSTM baseline for comparison."""
    
    def __init__(self, seq_len: int = 50, pred_len: int = 10, hidden_size: int = 64,
                 num_layers: int = 2, epochs: int = 50, lr: float = 0.001):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_model(self):
        """Create LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])  # Use last timestep
                return out
        
        return LSTMModel(1, self.hidden_size, self.num_layers, self.pred_len)
    
    def fit(self, series: np.ndarray):
        """Fit LSTM model to time series."""
        try:
            if len(series) < self.seq_len + self.pred_len + 10:
                return False
            
            # Normalize data
            series_scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
            
            # Create dataset
            dataset = TimeSeriesDataset(series_scaled, self.seq_len, self.pred_len)
            if len(dataset) == 0:
                return False
            
            dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            # Initialize model
            self.model = self._create_model().to(self.device)
            
            # Training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
            self.model.train()
            for epoch in range(self.epochs):
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.unsqueeze(-1).to(self.device)  # Add feature dim
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.last_sequence = series_scaled[-self.seq_len:]
            return True
            
        except Exception as e:
            print(f"LSTM fitting failed: {e}")
            return False
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate LSTM forecasts."""
        if self.model is None:
            return np.array([np.nan] * steps)
        
        try:
            self.model.eval()
            predictions = []
            current_seq = self.last_sequence.copy()
            
            with torch.no_grad():
                remaining_steps = steps
                
                while remaining_steps > 0:
                    input_tensor = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                    pred = self.model(input_tensor).cpu().numpy().flatten()
                    
                    steps_to_take = min(remaining_steps, len(pred))
                    predictions.extend(pred[:steps_to_take])
                    
                    if remaining_steps > len(pred):
                        current_seq = np.concatenate([current_seq[len(pred):], pred])
                    
                    remaining_steps -= steps_to_take
            
            # Denormalize
            predictions = np.array(predictions[:steps])
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            return predictions
            
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            return np.array([np.nan] * steps)

class TransformerModelEvaluator:
    """Evaluates transformer models on time series datasets."""
    
    def __init__(self):
        self.models = {
            'Transformer': TransformerForecaster,
            'Large_Transformer': LargeTransformerForecaster,
            'Decoder_Only': DecoderOnlyForecaster,
            'LSTM': SimpleLSTMForecaster
        }
        self.results = []
    
    def evaluate_single_series(self, train_data: np.ndarray, test_data: np.ndarray,
                              series_id: str, dataset_name: str) -> Dict[str, Any]:
        """Evaluate transformer models on a single time series."""
        
        series_results = {
            'series_id': series_id,
            'dataset': dataset_name,
            'train_length': len(train_data),
            'test_length': len(test_data)
        }
        
        for model_name, model_class in self.models.items():
            print(f"    {model_name}...", end=' ')
            
            try:
                # Initialize model with appropriate parameters
                pred_len = min(len(test_data), 10)  # Predict up to 10 steps
                seq_len = min(50, len(train_data) // 2)  # Use up to 50 steps of history
                
                # Handle different model parameter requirements
                if model_name == 'Decoder_Only':
                    # DecoderOnlyForecaster doesn't use pred_len
                    model = model_class(
                        seq_len=seq_len,
                        epochs=20  # Reduced for faster training
                    )
                else:
                    model = model_class(
                        seq_len=seq_len,
                        pred_len=pred_len,
                        epochs=20  # Reduced for faster training
                    )
                
                # Fit model
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
                if not np.any(np.isnan(predictions)) and len(predictions) == len(test_data):
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
                    print("Invalid predictions")
                    
            except Exception as e:
                series_results[f'{model_name}_mae'] = np.nan
                series_results[f'{model_name}_rmse'] = np.nan
                series_results[f'{model_name}_predictions'] = [np.nan] * len(test_data)
                print(f"ERROR: {str(e)[:50]}...")
        
        return series_results
    
    def evaluate_dataset(self, dataset_path: str, max_series: int = 5) -> List[Dict[str, Any]]:
        """Evaluate transformer models on dataset (limited for speed)."""
        print(f"\nEvaluating transformer models on: {dataset_path}")
        
        # Load data
        metadata_file = dataset_path.replace('_values.npz', '_processed.csv')
        values_file = dataset_path
        
        metadata_df = pd.read_csv(metadata_file)
        values_data = np.load(values_file, allow_pickle=True)
        
        train_values = values_data['train_values']
        test_values = values_data['test_values']
        
        dataset_results = []
        
        # Limit to first few series for initial testing
        n_series = min(max_series, len(metadata_df))
        
        for i in range(n_series):
            row = metadata_df.iloc[i]
            print(f"  Series {i+1}/{n_series}: {row['series_id']}")
            
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
    """Main evaluation pipeline for transformer models."""
    print("="*60)
    print("TRANSFORMER MODELS EVALUATION")
    print("="*60)
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    evaluator = TransformerModelEvaluator()
    all_results = []
    
    # Evaluate on datasets (limited series for speed)
    datasets = [
        'data/trend_seasonal_values.npz',
        'data/multi_seasonal_values.npz', 
        'data/random_walk_values.npz'
    ]
    
    for dataset_path in datasets:
        try:
            results = evaluator.evaluate_dataset(dataset_path, max_series=3)
            all_results.extend(results)
        except Exception as e:
            print(f"Failed to evaluate {dataset_path}: {e}")
    
    # Save combined results
    if all_results:
        evaluator.save_results(all_results, 'results/transformer_models_results.csv')
        
        # Print summary statistics
        df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for model in ['Transformer', 'Large_Transformer', 'LSTM']:
            mae_col = f'{model}_mae'
            if mae_col in df.columns:
                mean_mae = df[mae_col].mean()
                valid_count = df[mae_col].notna().sum()
                print(f"{model:12} | Mean MAE: {mean_mae:.3f} | Valid: {valid_count}/{len(df)}")
    
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
