import torch
import torch.nn as nn

class TransformerParkinsonsModel(nn.Module):
    def __init__(self, input_size, num_classes, num_heads=8, num_layers=4, hidden_dim=128, dropout=0.1):
        super(TransformerParkinsonsModel, self).__init__()

        # 입력 임베딩을 위한 선형 변환
        self.input_embedding = nn.Linear(input_size, hidden_dim)

        # Positional Encoding 추가 (Transformer는 순서 정보를 필요로 하기 때문)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        # Transformer Encoder Layer 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4
        )

        # Transformer Encoder 구성
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification을 위한 최종 선형 레이어
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x의 형태는 (batch_size, sequence_length, input_size) 이어야 합니다.
        # Transformer의 입력 형태는 (sequence_length, batch_size, hidden_dim)이므로 변환 필요
        x = self.input_embedding(x)  # (batch_size, sequence_length, hidden_dim)
        x = x.permute(1, 0, 2)       # (sequence_length, batch_size, hidden_dim)

        # Positional Encoding 추가
        x = self.positional_encoding(x)

        # Transformer Encoder 적용
        x = self.transformer_encoder(x)

        # 최종 sequence의 평균을 사용해 classification
        x = x.mean(dim=0)  # (batch_size, hidden_dim)
        x = self.fc(x)      # (batch_size, num_classes)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional Encoding 생성
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
