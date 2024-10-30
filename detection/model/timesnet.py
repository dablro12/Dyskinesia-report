# %% [TimesNet] Classifier
from bayes_opt import BayesianOptimization
from model.timesnet_lite.timesnet.timesnet import TimesNet
from metrics.metrics import auc_score, recall_score
class Classifier(TimesNet):
    def __init__(self,  seq_len, stride, lr, epochs, batch_size, epoch_steps, prt_steps, device, pred_len, e_layers, d_model, d_ff, dropout, top_k, c_out, num_kernels, verbose, random_state):
        self.seq_len = seq_len
        self.stride = stride
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.prt_steps = prt_steps
        self.device = device
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.top_k = top_k
        self.c_out = c_out
        self.num_kernels = num_kernels
        self.verbose = verbose
        self.random_state = random_state
            
        # TimesNet 모델 초기화
        self.clf = TimesNet(
            seq_len=self.seq_len,
            stride=self.stride,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            epoch_steps=self.epoch_steps,
            prt_steps=self.prt_steps,
            device=self.device,
            pred_len=self.pred_len,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            top_k=self.top_k,
            c_out = self.c_out,
            num_kernels = self.num_kernels,
            verbose=self.verbose,
            random_state=self.random_state
        )

        # Bayesian Optimization 파라미터 설정
        self.pbounds = {'num_kernels': (1, 6)}
        self.optimizer = None

    def objective_function(self, num_kernels):
        # 최적화할 목표 함수 정의
        self.clf.num_kernels = int(num_kernels)
        losses = self.clf.fit(self.X_train)
        scores = self.clf.decision_function(self.X_train)
        
        roc_auc = auc_score(self.labels, scores)# AUC Score 계산
        y_pred = (scores >= 0.5).astype(int)
        # Recall Score 계산
        recall = recall_score(self.labels, y_pred)
        
        # AUC Score + Recall Score Weighted Sum 
        combined_score = 0.7 * roc_auc + 0.3 * recall # recall_weight = 0.3, auc_weight = 0.7

        # 목표 함수 값 반환
        return combined_score
    
    def optimize(self, X_train, labels, init_points=3, n_iter=3):
        # 데이터 셋팅
        self.X_train = X_train
        self.labels = labels
        
        # Bayesian Optimization 초기화
        self.optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.pbounds,
            verbose=2,
            random_state=42
        )

        # 최적화 실행
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # 최적의 커널 개수 및 목적 함수 값 출력
        best_params = self.optimizer.max['params']
        best_target = self.optimizer.max['target']
        
        # 반올림 처리
        best_num_kernels = int(round(best_params['num_kernels']))
        print("\033[48;5;230m" + f"\033[38;5;214m최적의 커널 개수 : {best_params}\033[0m")
        print("\033[48;5;230m" + f"\033[38;5;214m최적의 목적 함수 값 : {best_target}\033[0m")
        print("\033[48;5;230m" + f"\033[38;5;214m결정된 커널 수 : {best_num_kernels}\033[0m")
        
        
        # 최적의 num_kernels 반환
        return int(best_params['num_kernels']), best_target, best_num_kernels

    def get_best_num_kernels_model(self, best_k_num):
        self.clf = TimesNet(
        seq_len=self.seq_len,
        stride=self.stride,
        lr=self.lr,
        epochs=self.epochs,
        batch_size=self.batch_size,
        epoch_steps=self.epoch_steps,
        prt_steps=self.prt_steps,
        device=self.device,
        pred_len=self.pred_len,
        e_layers=self.e_layers,
        d_model=self.d_model,
        d_ff=self.d_ff,
        dropout=self.dropout,
        top_k=self.top_k,
        c_out = self.c_out,
        num_kernels = best_k_num,
        verbose=self.verbose,
        random_state=self.random_state
        )
        
        return self.clf

