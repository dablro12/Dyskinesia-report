import torch 
from model.timesnet import Classifier
# CUDA 사용 가능 여부에 따라 GPU 또는 CPU 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(X_train, labels, config, bayes_opt=False):
    """
    모델 초기화 및 최적 커널 수 검색 후 최적화된 모델 반환 함수
    """
    # TimesNetOptimizer 초기화
    kernel_optimizer = Classifier(
        seq_len=config['seq_len'],           # 시퀀스 길이: 입력 시계열 데이터의 길이 설정
        stride=config['stride'],               # 이동 간격: 각 시퀀스를 슬라이딩 윈도우 방식으로 학습
        lr=config['lr'],              # 학습률: 학습 속도 설정
        epochs=config['epochs'],               # 에폭 수: 학습 반복 횟수
        batch_size=config['batch_size'],         # 배치 크기: 학습에 사용할 배치의 크기 설정
        epoch_steps=config['epoch_steps'],         # 각 에폭당 최대 반복 수: 에폭 당 학습 단계 수 제한
        prt_steps=config['prt_steps'],            # 출력 단계: 출력할 학습 단계 간격
        device=config['device'],          # 디바이스 설정: GPU 또는 CPU 사용
        pred_len=config['pred_len'],             # 예측 길이: 출력으로 생성할 시계열 예측 길이
        e_layers=config['e_layers'],             # 인코더 레이어 수: 시계열 패턴 학습에 사용될 레이어 수
        d_model=config['d_model'],             # 모델 차원: 임베딩 및 학습에 사용될 차원 수
        d_ff=config['d_ff'],                # FFN 레이어 차원: 피드포워드 레이어의 차원
        dropout=config['dropout'],            # 드롭아웃 비율: 과적합 방지를 위한 드롭아웃 비율
        top_k=config['top_k'],                  # 최상위 필터 수: 가장 중요한 상위 k개의 필터를 선택
        c_out=config['c_out'],                  # 출력 채널 수: 합성곱 레이어의 출력 채널 수
        num_kernels=config['num_kernels'],          # 커널 수: 합성곱 필터의 수
        verbose=config['verbose'],              # 출력 설정: 학습 과정에서 출력할 상세 정보의 레벨
        random_state=config['random_state'],         # 랜덤 시드: 결과 재현성을 위한 랜덤 시드
    )

    # 최적화 실행
    if bayes_opt:
        _, _, best_k_num = kernel_optimizer.optimize(X_train, labels, init_points=3, n_iter=3)
    else:
        _, _, best_k_num = None, None, config['num_kernels']  # 최적 커널 수: 최적 커널 수를 직접 설정
    
    config['num_kernels'] = best_k_num  # 최적 커널 수 저장

    # 최적 커널 수로 초기화된 최적 모델을 가져옴
    best_clf = kernel_optimizer.get_best_num_kernels_model(best_k_num)  # 베이지안 최적화 결과로 얻은 최적 커널 수 모델
    best_clf.set_seed(config["random_state"])

    return best_clf, config

# %% Using
# model = init_model(X_train, y_train) # Best Kernel Optimization 
# model = trainer(model, X_train) # Unsupervised Learning
# model, metric_dict, adj_metric_dict = tester(model, X_test, y_test) # Test 