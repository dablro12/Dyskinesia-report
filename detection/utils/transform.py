import cv2

def resize_module(data, size =640):
    return cv2.resize(data, (size,size))

import torch
def reshape_tensor(data_loader):
    x_train_arr, y_train_arr = [], []
    for X, y in data_loader:
        # X shape (batch, seq_len, feature) -> (batch * seq_len, feature)
        X = X.view(-1, X.shape[-1])
        # Y shape (batch, seq_len) - > (batch * seq_len)
        y = y.view(-1)
        
        x_train_arr.append(X)
        y_train_arr.append(y)
    # X_train_arr shape (iter, batch * seq_len, feature) -> (iter * batch * seq_len, feature)
    x_train_arr = torch.cat(x_train_arr, dim=0)
    y_train_arr = torch.cat(y_train_arr, dim=0)
    
    x_train_arr = x_train_arr.cpu().numpy()
    y_train_arr = y_train_arr.cpu().numpy()
    
    return x_train_arr, y_train_arr

# %% X-mark-encoder 변수로써 시간을 넣기 위한 함수
import torch
def create_time_features(batch_size, seq_len):
    # 각 프레임에 대한 시간 순서 생성 (0부터 seq_len-1까지)
    time_seq = torch.arange(seq_len).float()
    # 배치 크기만큼 반복
    time_seq = time_seq.repeat(batch_size, 1)
    return time_seq

# # 사용 예시
# batch_size, seq_len = 48, 600
# x_mark_enc = create_time_features(batch_size, seq_len)

# %% multi class 
# def reshape_multi_tensor(data_loader):
#     x_train_arr, y_train_arr = [], [] 
#     for X, y in data_loader:
#         X = X.view(-1, X.shape[-1])
        
#         x_train_arr.append(X)
#         y_train_arr.append(y)
#     print(x_train_arr.shape)
#     x_train_arr = torch.cat(x_train_arr, dim=0)
#     y_train_arr = torch.cat(y_train_arr, dim=0)
    
#     x_train_arr = x_train_arr.cpu().numpy()
#     y_train_arr = y_train_arr.cpu().numpy()
    
#     return x_train_arr, y_train_arr
def reshape_multi_tensor(data_loader):
    x_train_arr, y_train_arr = [], [] 
    for X, y in data_loader:
        # X의 크기 확인
        # X를 reshape하여 배치 크기와 시퀀스 길이를 유지
        batch_size = X.shape[0]
        X = X.view(batch_size * X.shape[1], X.shape[-1])  # (배치 크기 * 600, 96)
        
        x_train_arr.append(X)
        y_train_arr.append(y)

    # 각 배열을 concatenate
    x_train_arr = torch.cat(x_train_arr, dim=0)
    y_train_arr = torch.cat(y_train_arr, dim=0)
    
    x_train_arr = x_train_arr.cpu().numpy()
    y_train_arr = y_train_arr.cpu().numpy()
    
    return x_train_arr, y_train_arr    