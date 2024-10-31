import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time
from model.timesnet_lite.base.base_model import BaseDeepAD
from model.timesnet_lite.base.utility import get_sub_seqs, get_sub_seqs_label
from .timesnetmodule import TimesBlock, DataEmbedding
import torch.nn.functional as F
from metrics.metrics import test_metric
from utils.transform import create_time_features

# %% TimesNet
class TimesNet(BaseDeepAD):
    def __init__(self, seq_len=100, stride=1, lr=0.0001, epochs=10, batch_size=32,
        epoch_steps=20, prt_steps=1, device='cuda',
        pred_len=0, e_layers=2, d_model=64, d_ff=64, dropout=0.1, top_k=5, c_out = 1, num_kernels=6,
        verbose=2, random_state=42, net=None):

        super(TimesNet, self).__init__(
            model_name='TimesNet', data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.c_out = c_out
        self.net = net
        
    def fit(self, X, y=None):
        self.n_features = X.shape[1] # 
        train_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        if self.c_out == 1: # for binary classification
            print(f"## Init Binary Classification DataLoad ##")
            train_labels = get_sub_seqs_label(y, seq_len=self.seq_len, stride=self.stride) # for anomaly detection
            dataloader = DataLoader(list(zip(train_seqs, train_labels)), batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True, num_workers=8)
        elif self.c_out > 1: # for multi-label class classification
            print(f"## Init Multi Label Classification DataLoad ##")
            dataloader = DataLoader(list(zip(train_seqs, y)), batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True, num_workers=8)
            
        self.net = TimesNetModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.n_features,
            c_out=self.c_out,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            top_k=self.top_k,
            num_kernels=self.num_kernels
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        train_losses = []
        for e in range(self.epochs):
            t1 = time.time()
            loss = self.training(dataloader)
            train_losses.append(loss)
            if self.verbose >= 1 and (e == 0 or (e + 1) % self.prt_steps == 0): # 만약 verbose가 1이상이고, e가 0이거나 prt_steps의 배수일 때
                print(f"epoch {e + 1:3d}, training loss: {loss:.6f}, time: {time.time() - t1:.1f}s")
                
        # self.decision_scores_ = self.decision_function(X, y) # 검증
        # self.labels_ = self._process_decision_scores()  # in base model
        return train_losses

    def decision_function(self, X, y, return_rep=False):
        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        if self.c_out == 1: # for binary classification
            print(f"## Init Binary Classification DataLoad ##")
            labels = get_sub_seqs_label(y, seq_len=self.seq_len, stride=self.stride)
            dataloader = DataLoader(list(zip(seqs, labels)), batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True, num_workers=8)
        elif self.c_out > 1: # for multi-label class classification
            print(f"## Init Multi Label Classification DataLoad ##")
            dataloader = DataLoader(list(zip(seqs, y)), batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True, num_workers=8)
        
        metric_dict = self.inference(dataloader)  # (n,d)
        return metric_dict
        # # print(f"[loss] : {loss.shape}")
        # loss_final = np.mean(loss, axis=1)  # (n,)

        # padding_list = np.zeros([X.shape[0] - loss.shape[0], loss.shape[1]])
        # loss_pad = np.concatenate([padding_list, loss], axis=0)
        # loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])

        return loss_final_pad


    def training(self, dataloader):
        if self.c_out == 1:
            criterion = nn.BCEWithLogitsLoss().to(self.device)  # for binary classification    
        elif self.c_out > 1:
            # criterion = nn.BCEWithLogitsLoss().to(self.device)  # for Multi-label class classification
            # criterion = nn.MultiLabelSoftMarginLoss().to(self.device) # for multi-label class classification
            criterion = FocalLoss(alpha=0.5, gamma=2).to(self.device)

        # criterion = nn.MSELoss().to(self.device)
        # criterion = nn.SmoothL1Loss().to(self.device)
        # criterion = nn.CrossEntropyLoss().to(self.device)  # for multi-label class classification
        train_loss = []
        self.net.train()
        print("\033[48;5;230m" + f"\033[38;5;115m ### TRAINING ### \033[0m")
        for ii, (batch_x, batch_y) in enumerate(dataloader): 
            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)  # (bs, seq_len, dim)
            batch_y = batch_y.float().to(self.device)  # (bs, seq_len, dim)
            # 시간 속성 넣기위해 x_mark_enc 생성
            x_mark_enc = create_time_features(batch_size = batch_x.shape[0], seq_len = batch_x.shape[1]).to(self.device)
            
            # Predict
            outputs = self.net(x_enc = batch_x, x_mark_enc = x_mark_enc)  # [B, N] for classification, [B, L, D] for anomaly detection

            # loss = criterion(outputs[:, -1:, :], batch_x[:, -1:, :]) # for anomaly detection
            if self.c_out == 1:
                batch_y = batch_y[:, -1].unsqueeze(1)
                loss = criterion(outputs, batch_y)  # output : B, N / batch_y : B, Freme*label
            elif self.c_out > 1:
                batch_y = batch_y[:, -1, :]
                loss = criterion(outputs, batch_y) # output

            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if self.epoch_steps != -1:
                if ii > self.epoch_steps:
                    break
        self.scheduler.step()

        return np.average(train_loss)

    def inference(self, dataloader):
        label_li, prob_li = [], []
        self.net.eval()
        print("\033[48;5;230m" + f"\033[38;5;115m ### EVALUATION ### \033[0m")
        with torch.no_grad():
            for batch_x, batch_y in dataloader:  # test_set
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)  # (bs, seq_len, dim)
                # 시간 속성 넣기위해 x_mark_enc 생성
                x_mark_enc = create_time_features(batch_size = batch_x.shape[0], seq_len = batch_x.shape[1]).to(self.device)
                outputs = self.net(x_enc = batch_x, x_mark_enc = x_mark_enc)  # [B, N] for classification, [B, L, D] for anomaly detection
                
                # criterion
                if self.c_out == 1:
                    batch_y = batch_y[:, -1].unsqueeze(1)
                    prob_li.append(torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy())
                elif self.c_out > 1:
                    batch_y = batch_y[:, -1, :]
                    prob = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
                    prob_li.append(prob)
                label_li.append(batch_y.squeeze(1).detach().cpu().numpy())

                # output_sample = F.softmax(outputs[0, :], dim = 0).detach().cpu().numpy() # 다중 클래ㅡㅅ
                output_sample = torch.sigmoid(outputs[0, :]).detach().cpu().numpy() # 다중 라벨 
                batch_y_sample = batch_y[0, :].detach().cpu().numpy()
                pred_sample = (output_sample > 0.5).astype(float)
                print(f"Label : {batch_y_sample}, \n Pred : {pred_sample}")
            if self.c_out == 1:
                label_arr = np.array(label_li).flatten()
                prob_arr = np.array(prob_li).flatten()
            elif self.c_out > 1: # 1, (B)24, (Pred Class)8 -> (B)24, (Pred Class)8 
                label_arr = np.array(label_li).squeeze(0).flatten()
                prob_arr = np.array(prob_li).squeeze(0).flatten()
            
            metric_dict = test_metric(label_arr, prob_arr, thr_li = [0.1, 0.25, 0.5, 0.75, 0.9], n_features = self.c_out)# Metric 
        return metric_dict

    def inference_prepare(self, X, weight):
        """define test_loader"""
        self.n_features = X.shape[1]
        infer_seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)
        dataloader = DataLoader(infer_seqs, batch_size=self.batch_size,
                                shuffle=True, pin_memory=True, num_workers=8)
        
        self.net = TimesNetModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.n_features,
            c_out=self.c_out,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            top_k=self.top_k,
            num_kernels=self.num_kernels
        ).to(self.device)
        
        if weight is not None:
            print("\033[48;5;230m" + f"\033[38;5;214m[Model WEIGTH LOAD COMPLETE]\033[0m")
            self.net.load_state_dict(weight)

        return dataloader
    
    def inference_forward(self, reshape_X, weight, thr):
        """define forward step in inference"""
        class_infer_loader = self.inference_prepare(X = reshape_X, weight = weight)
        self.net.eval()
        
        prediction_li = []
        with torch.no_grad():
            for idx, batch_x in enumerate(class_infer_loader):
                batch_x = batch_x.float().to(self.device)
                x_mark_enc = create_time_features(batch_size = batch_x.shape[0], seq_len = batch_x.shape[1]).to(self.device)
                outputs = self.net(x_enc = batch_x, x_mark_enc = x_mark_enc)
                # print(f"batch_x : {batch_x.shape}, x_mark_enc : {x_mark_enc.shape}, outputs : {outputs.shape}") # batch_x : torch.Size([1, 600, 96]), x_mark_enc : torch.Size([1, 600]), outputs : torch.Size([1, 1])
                
                if self.c_out == 1: # for binary classification
                    prob = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
                    pred = (prob >= thr).astype(int)
                elif self.c_out > 1: # for multi-label class classification
                    prob = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
                    pred = (prob >= thr).astype(int)
                    
                prediction_li.append(pred)
        # (iteration, B, Feature) -> (iteration*B, Feature)
        prediction_li = np.vstack(prediction_li)  # (배치 수, 피쳐 수)
        return prediction_li
    
    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

# %% Parkinson Dyskniesia Classification Model
class TimesNetModel(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, c_out,
                 e_layers, d_model, d_ff, dropout, top_k, num_kernels):

        super(TimesNetModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels) for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.c_out = c_out
        # self.projection = nn.Linear(d_model, c_out, bias=True) # for anomaly detection

        # classifcation task 
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        # binary classificaiton
        # self.projection = nn.Linear(
        #     in_features = d_model * seq_len,
        #     out_features= c_out
        # )
        self.projection = nn.Sequential(
            # Shallow Classifier
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, c_out)
            # Deep Classifier
            # nn.Linear(d_model, d_model // 2),
            # nn.ReLU(),
            # nn.Linear(d_model // 2, (d_model // 2) // 2),
            # nn.ReLU(),
            # nn.Linear((d_model // 2) // 2, c_out)
            
        )

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)

        # zero-out padding embeddings:The primary role of x_mark_enc in the code is to 
        # zero out the embeddings for padding positions in the output tensor through 
        # element-wise multiplication, helping the model to focus on meaningful data 
        # while disregarding padding.
        output = output * x_mark_enc.unsqueeze(-1)
        
        # (batch_size, seq_length * d_model)
        # output = output.reshape(output.shape[0], -1)
        # output = self.projection(output)  # (batch_size, num_classes)
        output = output.mean(dim=1)  # sequence 차원을 평균내서 차원 축소
        output = self.projection(output)
        return output    
    def forward(self, x_enc, x_mark_enc):
        # dec_out = self.anomaly_detection(x_enc) # for anomaly detection
        # return dec_out  # [B, L, D]
        dec_out = self.classification(x_enc, x_mark_enc) # for classification
        return dec_out  # [B, N] return the classification result
    
    
# %% Focal Loss for Multi-label Classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sigmoid로 확률을 구함
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # 예측 확률
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss