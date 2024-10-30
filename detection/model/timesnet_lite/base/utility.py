import numpy as np
import torch 

def get_sub_seqs(x_arr, seq_len=600, stride=1):
    seq_starts = np.arange(0, x_arr.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    # 수정된 코드
    # x_seqs = np.array(
    #     [x_arr[i:i + seq_len].numpy() if isinstance(x_arr[i:i + seq_len], torch.Tensor) else x_arr[i:i + seq_len]
    #     for i in seq_starts], dtype=np.float32
    # )
    return x_seqs


def get_sub_seqs_label(y, seq_len=600, stride=1):
    seq_starts = np.arange(0, y.shape[0] - seq_len + 1, stride)
    y_seq = np.array([y[i:i + seq_len] for i in seq_starts])
    # y = np.sum(y_seq, axis=1) / seq_len

    # y_binary = np.zeros_like(y)
    # y_binary[np.where(y != 0)[0]] = 1
    return y_seq
