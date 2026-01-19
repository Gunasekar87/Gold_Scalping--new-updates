import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    """
    Institutional-Grade Transformer for Market Prediction.
    Uses Multi-Head Self-Attention to detect complex temporal patterns 
    across price, volume, and volatility.
    """
    def __init__(self, input_dim=5, order_book_dim=40, d_model=128, nhead=4, num_layers=2, output_dim=3, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.model_type = 'Transformer'
        self.d_model = d_model

        # 1. Feature Embedding (Time Series)
        # Projects raw inputs (Open, High, Low, Close, Vol) into vector space
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        # Injects information about the relative or absolute position of the tokens in the sequence
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        # The "Brain" that figures out relationships between different time steps
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Order Book Encoder (The "Vision" Upgrade)
        # Processes the Level 2 Snapshot (Bids/Asks)
        # Input: [batch_size, depth * 2] (Flattened Price/Vol pairs)
        self.ob_encoder = nn.Sequential(
            nn.Linear(order_book_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # 5. Output Heads
        # Combined Dimension = d_model (Time Series Context) + 32 (Order Book Context)
        combined_dim = d_model + 32
        
        # Head A: Trend Prediction (Buy/Sell/Neutral)
        self.trend_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Softmax applied in loss function
        )
        
        # Head B: Volatility Prediction (ATR / Range)
        self.volatility_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Regression output
        )

    def forward(self, src, ob_src=None):
        """
        src shape: [batch_size, seq_len, input_dim]
        ob_src shape: [batch_size, order_book_dim] (Optional)
        """
        # Embed and add position info
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through Transformer
        # Output shape: [batch_size, seq_len, d_model]
        output = self.transformer_encoder(src)
        
        # We only care about the LAST time step (the most recent candle) for prediction
        # Take the last vector in the sequence
        last_step = output[:, -1, :]
        
        # Process Order Book if provided
        if ob_src is not None:
            ob_embedding = self.ob_encoder(ob_src)
            # Concatenate Time Series Context + Order Book Context
            combined_features = torch.cat((last_step, ob_embedding), dim=1)
        else:
            # If no Order Book data (e.g., during pre-training on simple data), pad with zeros
            dummy_ob = torch.zeros(last_step.size(0), 32, device=last_step.device)
            combined_features = torch.cat((last_step, dummy_ob), dim=1)
        
        # Predictions
        trend_logits = self.trend_head(combined_features)
        volatility_pred = self.volatility_head(combined_features)
        
        return trend_logits, volatility_pred

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
