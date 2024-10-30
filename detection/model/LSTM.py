import torch
import torch.nn as nn
# LSTM 모델 클래스 정의
class LSTMParkinsonsModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMParkinsonsModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어 정의
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer 정의 (LSTM의 출력 -> 최종 예측값)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # LSTM 레이어 적용
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 cell state
        
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM 적용
        out = out[:, -1, :]  # 시퀀스의 마지막 time step의 결과를 사용
        
        # Fully connected 레이어 적용
        out = self.fc(out)
        return out
