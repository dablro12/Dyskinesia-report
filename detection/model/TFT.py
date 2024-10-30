import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_dim, num_heads, num_layers, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()

        # 1. 입력 임베딩
        self.input_embedding = nn.Linear(input_size, hidden_dim)
        
        # 2. LSTM 계층
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # 3. 멀티헤드 어텐션
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 4. 게이트 메커니즘을 위한 선형 레이어
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        # 5. 출력 레이어
        self.fc = nn.Linear(hidden_dim, 1)  # 이진 분류를 위한 출력층
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 입력 임베딩
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM 계층
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # 멀티헤드 어텐션
        # lstm_out의 형태가 (batch_size, seq_len, hidden_dim)이어야 하므로 permute 불필요
        attn_output, _ = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        
        # 게이트 메커니즘 적용
        concatenated = torch.cat((lstm_out, attn_output), dim=-1)  # (batch_size, seq_len, hidden_dim*2)
        gate_values = self.sigmoid(self.gate(concatenated))  # (batch_size, seq_len, hidden_dim)
        gated_output = gate_values * lstm_out + (1 - gate_values) * attn_output
        
        # 최종 출력
        output = self.fc(self.dropout(gated_output[:, -1, :]))  # 마지막 타임스텝의 출력 사용
        
        return output
