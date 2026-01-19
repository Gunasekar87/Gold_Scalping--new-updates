
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import pandas as pd
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from .nexus_transformer import TimeSeriesTransformer

# Setup logging
logger = logging.getLogger("NexusTrainer")

class NexusTrainer:
    """
    Trainer for the Nexus TimeSeriesTransformer.
    Handles data loading, preprocessing, and the training loop.
    """
    def __init__(self, db_path="data/market_memory.db", model_save_path="models/nexus_transformer.pth"):
        self.db_path = db_path
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.seq_len = 64
        self.batch_size = 32
        self.d_model = 128
        self.learning_rate = 0.001
        
        # Scaling factors (approximate for Gold)
        self.price_scale = 5000.0 
        self.vol_scale = 1000.0

    def load_data(self):
        """
        Load M1 candles from SQLite database.
        Returns DataFrame or None if empty.
        """
        if not os.path.exists(self.db_path):
            return None
            
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT open, high, low, close, volume FROM candles WHERE timeframe='M1' ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return None
                
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None

    def create_sequences(self, data):
        """
        Convert dataframe to sequences (sliding window).
        Data format: [Open, High, Low, Close, Volume]
        """
        sequences = []
        targets_trend = []
        targets_vol = []
        
        values = data.values.astype(float)
        
        # Normalize
        values[:, :4] /= self.price_scale # Normalize OHLC
        values[:, 4] = np.log1p(values[:, 4]) # Log normalize volume
        
        for i in range(len(values) - self.seq_len - 1):
            seq = values[i : i + self.seq_len]
            
            # Target: Next Close vs Current Close (0=Down, 1=Neutral, 2=Up)
            current_close = values[i + self.seq_len - 1, 3]
            next_close = values[i + self.seq_len, 3]
            diff = next_close - current_close
            
            if diff > 0.0001: # Small threshold
                trend = 2 # Up
            elif diff < -0.0001:
                trend = 0 # Down
            else:
                trend = 1 # Neutral
                
            # Target Volatility (Next High - Next Low)
            volatility = values[i + self.seq_len, 1] - values[i + self.seq_len, 2]
            
            sequences.append(seq)
            targets_trend.append(trend)
            targets_vol.append(volatility)
            
        return np.array(sequences), np.array(targets_trend), np.array(targets_vol)

    def generate_synthetic_data(self):
        """Generate synthetic sine wave data for initialization."""
        logger.info("Generating synthetic training data...")
        t = np.linspace(0, 100, 1000)
        price = 2000 + 10 * np.sin(t) + np.random.normal(0, 0.5, 1000)
        volume = 100 + 10 * np.abs(np.cos(t)) + np.random.normal(0, 5, 1000)
        
        df = pd.DataFrame({
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price, # Simple OHLC
            'volume': volume
        })
        return df

    def train(self, epochs=5):
        """Run training loop."""
        logger.info(f"Starting training on {self.device}...")
        
        # 1. Load Data
        df = self.load_data()
        if df is None or len(df) < self.seq_len + 10:
            logger.warning("Insufficient usage data. Using synthetic data for robust initialization.")
            df = self.generate_synthetic_data()
            
        # 2. Preprocess
        X, y_trend, y_vol = self.create_sequences(df)
        
        # Convert to Tensor
        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.LongTensor(y_trend).to(self.device),
            torch.FloatTensor(y_vol).to(self.device)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 3. Model Setup
        model = TimeSeriesTransformer(
            input_dim=5, 
            d_model=self.d_model
        ).to(self.device)
        
        criterion_trend = nn.CrossEntropyLoss()
        criterion_vol = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        model.train()
        total_loss = 0.0
        
        # 4. Loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            steps = 0
            for batch_X, batch_y_trend, batch_y_vol in loader:
                optimizer.zero_grad()
                
                # Forward
                pred_trend, pred_vol = model(batch_X)
                
                # Loss
                loss_t = criterion_trend(pred_trend, batch_y_trend)
                loss_v = criterion_vol(pred_vol.squeeze(), batch_y_vol)
                loss = loss_t + loss_v
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
                
            avg_loss = epoch_loss / max(1, steps)
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            total_loss = avg_loss

        # 5. Save
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), self.model_save_path)
        logger.info(f"Model saved to {self.model_save_path}")
        
        return total_loss
