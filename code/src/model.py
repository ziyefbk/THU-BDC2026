import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeatureWiseAttention(nn.Module):
    """Attention across feature dimension for each (batch*num_stocks, seq_len, d_model)."""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(FeatureWiseAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.norm(x + self.out_proj(out))
        return out


class CrossStockGAT(nn.Module):
    """GAT-style cross-stock attention with learnable edge (pairwise) weights."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockGAT, self).__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_proj = nn.Linear(d_model, nhead)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, stock_features):
        B, N, D = stock_features.shape

        q = self.q_proj(stock_features).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(stock_features).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(stock_features).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        edge_scores = self.edge_proj(stock_features).view(B, N, self.nhead)
        edge_bias = (torch.sigmoid(edge_scores) + 0.5).unsqueeze(-1)
        edge_bias = edge_bias.permute(0, 2, 1, 3)
        attn_logits = attn_logits + edge_bias.log()

        attn = F.leaky_relu(attn_logits, 0.2)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.norm(stock_features + self.out_proj(out))
        return out


class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=32):
        super(StockTransformer, self).__init__()
        self.model_type = 'EnhancedStockTransformer'
        self.config = config
        self.num_stocks = num_stocks

        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        self.feature_attention = FeatureWiseAttention(config['d_model'], num_heads=4, dropout=config['dropout'])

        self.cross_stock_attention = CrossStockGAT(config['d_model'], config['cross_nhead'], config['dropout'])

        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )

        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.GELU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src):
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        temporal_features = self.temporal_encoder(src_proj)

        attended_features = self.feature_attention(temporal_features)

        agg = attended_features.mean(dim=1)
        stock_features = agg.view(batch_size, num_stocks, -1)

        interactive_features = self.cross_stock_attention(stock_features)

        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        ranking_features = self.ranking_layers(interactive_features)
        scores = self.score_head(ranking_features)
        output = scores.view(batch_size, num_stocks)

        return output
