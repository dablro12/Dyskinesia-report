# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# %% [TimesNet] Train / Test Code Runner
import torch 
import argparse
from tqdm import tqdm
config = {
        "seq_len":600,  # 시퀀스 길이: 입력 시계열 데이터의 길이 설정 : 20초 영상 30fps 기준 600 프레임을 seq_len으로 설정
        "stride":600,      # 이동 간격: 각 시퀀스를 슬라이딩 윈도우 방식으로 학습 : 20초 영상 30fps 기준 600 프레임을 stride로 설정
        "lr":0.001,     # 학습률: 학습 속도 설정
        "epochs":200,      # 에폭 수: 학습 반복 횟수
        "batch_size":48,# 배치 크기: 학습에 사용할 배치의 크기 설정
        "epoch_steps":1,# 각 에폭당 최대 반복 수: 에폭 당 학습 단계 수 제한
        "prt_steps":1,   # 출력 단계: 출력할 학습 단계 간격
        "pred_len":0,    # 예측 길이: 출력으로 생성할 시계열 예측 길이
        "e_layers":4,    # 인코더 레이어 수: 시계열 패턴 학습에 사용될 레이어 수
        "d_model":256,    # 모델 차원: 임베딩 및 학습에 사용될 차원 수
        "d_ff":256,       # FFN 레이어 차원: 피드포워드 레이어의 차원 
        "dropout":0.1,   # 드롭아웃 비율: 과적합 방지를 위한 드롭아웃 비율
        "top_k":5,       # 최상위 필터 수: 가장 중요한 상위 k개의 필터를 선택
        "num_kernels":3, # 커널 수: 합성곱 필터의 수
        "verbose":2,     # 출력 설정: 학습 과정에서 출력할 상세 정보의 레벨
        "random_state":42,         # 랜덤 시드: 결과 재현성을 위한 랜덤 시드
        "c_out" : 8,     # 이상 운동증 분류 모델 : 1 / 이상 운동증 증상 분류 모델 : 8
        "task_type" : "multi_classification", # data type 
        "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"), # 디바이스 설정: GPU 또는 CPU 사용
}

if __name__ == "__main__":
    from model import init_model
    from script.trainer import trainer
    from script.tester import tester
    from dataset.custom_dataset import init_dataloader
    from utils.transform import reshape_tensor, reshape_multi_tensor
    from model.params import get_model_size
    from utils.plotter import loss_plotter, metric_plotter
    
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--data_type', type=str, help='data type')
    argparse.add_argument('--data_dir', type =str, help='data directory')
    argparse.add_argument('--fold_dir', type=str, help='fold directory')
    argparse.add_argument('--save_dir', type=str, help='save directory')
        
    args = argparse.parse_args()

    config['data_type'] = args.data_type
    config['data_dir'] = args.data_dir
    # if config['task_type'] == 'binary_classification':
    #     config["c_out"] = 1,     # 출력 채널 수: 출력 시계열 데이터의 채널 수
    # else: # for Multi Class Classification
    #     config["c_out"] = 96,     # 출력 채널 수: 출력 시계열 데이터의 채널 수
    
    # Configuration Visual
    print("\033[48;5;230m" + f"\033[38;5;214m[Configuration] : {config}\033[0m")    
    
    for k_num in tqdm(range(1, 5), desc='Fold Loop'):
        train_loader, test_loader = init_dataloader(
            data_dir=os.path.join(config['data_dir'], config['data_type']),
            fold_dir=args.fold_dir,
            fold_num=k_num,
            type=config['task_type']
        )
        if config['c_out'] == 1:
            X_train, y_train = reshape_tensor(train_loader)
            X_test, y_test = reshape_tensor(test_loader)
        elif config['c_out'] == 8:
            X_train, y_train = reshape_multi_tensor(train_loader)
            X_test, y_test = reshape_multi_tensor(test_loader)
        # 주황색 텍스트, 밝은 배경색 적용
        print("\033[48;5;230m" + f"\033[38;5;214m[Train] : {X_train.shape, y_train.shape}\033[0m")
        print("\033[48;5;230m" + f"\033[38;5;214m[Test] : {X_test.shape, y_test.shape}\033[0m")
        # Model Init * 파라미터 수정은 model/__init__.py에서 수정 * 만약 최적화 하고 싶다면 bayes_opt = True로 설정
        model, config = init_model(X_train, y_train, config, bayes_opt= False) # Best Kernel Optimization 
        
        # Train  
        # model, train_losses = trainer(model, X_train) # Unsupervised Learning for anomaly detection
        model, train_losses = trainer(model, X_train, y_train) # Supervised Learning for classification
        # Test
        model, best_metric_dict = tester(model, X_test, y_test) # Test 

        # Save Loss & Model
        if config['task_type'] == 'binary_classification':
            loss_plotter(train_losses, save_path = os.path.join(args.save_dir, f"{config['data_type']}_{config['task_type']}_{k_num}_loss.png"))
            metric_plotter(best_metric_dict, save_path = os.path.join(args.save_dir, f"{config['data_type']}_{config['task_type']}_{k_num}_metric.png"))
        elif config['task_type'] == 'multi_classification':
            loss_plotter(train_losses, save_path = os.path.join(args.save_dir, f"{config['data_type']}_{config['task_type']}_{k_num}_loss.png"))
            metric_plotter(best_metric_dict, save_path = os.path.join(args.save_dir, f"{config['data_type']}_{config['task_type']}_{k_num}_metric.png"))
        save_path = os.path.join(args.save_dir, f"{config['data_type']}_{config['task_type']}_{k_num}.pth")
        torch.save({
            'model_state_dict': model.net.state_dict(),
            'model_size' : get_model_size(model),
            "config" : config,
            'metric_dict': best_metric_dict,
            "best_threshold" : best_metric_dict['best_thr']
        }, save_path)
        print("\033[48;5;250m" + f"\033[30m[Fold {k_num} Save Path] : {save_path}\033[0m")