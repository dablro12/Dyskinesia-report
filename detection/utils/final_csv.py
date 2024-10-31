    
import pandas as pd 
import numpy as np 

POSE_GUIDE = {
    "foot_left": [27, 29, 31],
    "foot_right": [28, 30, 32],
    "arm_left": [11, 13, 15],
    "arm_right": [12, 14, 16],
    "lower_body_right": [24, 26, 28],
    "lower_body_left": [23, 25, 27],
    "upper_body_right": [12, 24, 26],
    "upper_body_left": [11, 23, 25]
}

def result_post_process(coor_df:pd.DataFrame, vcc_acc_combined_df:pd.DataFrame, pred:np.array, POSE_GUIDE:dict, window_size:int=600):
    """ 
        pred : np.array, shape = ((All_Frame - Windows)/Stride + 1, 8)
    """
    # 2번째 칼럼부터 가져오기 
    coor_df = coor_df.iloc[:, 1:] 
    # 모든 칼럼에서 pose_ 라는 단어 제거
    coor_df.columns = [col.replace('pose_0_landmark_', '') for col in coor_df.columns]
    # 공백 값 채우기 
    coor_df = coor_df.fillna(method='ffill').fillna(method='bfill')
    
    # vcc_acc_mean 가져오기
    vcc_acc_mean_df = vcc_acc_mean_process(vcc_acc_combined_df, POSE_GUIDE)
    
    final_result_df = pd.concat([coor_df, vcc_acc_mean_df], axis=1)
    
    # pred를 데이터프레임으로 전환하는데 POSE_GUIDE의 인덱스
    pred_df = pred_to_df(pred, POSE_GUIDE, window_size= window_size)
    
    # final_result_df와 pred_df를 합치기
    final_result_df = pd.concat([final_result_df, pred_df], axis=1)

    return final_result_df

def pred_to_df(pred: np.array, POSE_GUIDE: dict, window_size: int = 600) -> pd.DataFrame:
    """
    pred의 shape는 (All_Frame - Windows)/Stride + 1, 8
    이때 8은 POSE_GUIDE의 개수와 같다. 
    pred를 데이터프레임으로 전환하는데 이때 칼럼은 POSE_GUIDE의 key값으로 한다.
    # 이때 예측되는 값의 row가 window_size번째부터 시작하므로 window_size부터 시작하도록 설정
    # 예를 들어 row 0부터 ~ (window_size - 1)까지는 NaN값으로 설정하고 그 뒤에 pred 값을 넣어준다. 
    """
    # NaN으로 채운 초기 데이터프레임 생성
    nan_df = pd.DataFrame(np.nan, index=range(window_size-1), columns=POSE_GUIDE.keys())
    
    # row index = window_size부터 업데이트
    pred_df = pd.DataFrame(pred, columns=POSE_GUIDE.keys())
    pred_df = pd.concat([nan_df, pred_df], ignore_index=True)
    
    return pred_df

def vcc_acc_mean_process(vcc_acc_combined_df: pd.DataFrame, POSE_GUIDE: dict) -> pd.DataFrame:
    # 각 증상에 대해 필요한 칼럼을 미리 생성
    all_velocity_cols = []
    all_acceleration_cols = []
    for coords in POSE_GUIDE.values():
        all_velocity_cols.extend([f"{coord}_{axis}_v" for coord in coords for axis in ['x', 'y', 'z']])
        all_acceleration_cols.extend([f"{coord}_{axis}_a" for coord in coords for axis in ['x', 'y', 'z']])

    # NumPy 배열로 변환
    data = vcc_acc_combined_df[all_velocity_cols + all_acceleration_cols].to_numpy()

    # 평균을 계산할 수 있는 인덱스 설정
    num_frames = data.shape[0]
    num_symptoms = len(POSE_GUIDE)

    results = np.zeros((num_frames, num_symptoms * 2))  # 결과를 저장할 배열
    symptom_names = []

    for i, (symptom_name, coords) in enumerate(POSE_GUIDE.items()):
        symptom_names.append(symptom_name)
        
        # 해당 증상에 대한 인덱스
        velocity_indices = [all_velocity_cols.index(f"{coord}_{axis}_v") for coord in coords for axis in ['x', 'y', 'z']]
        acceleration_indices = [all_acceleration_cols.index(f"{coord}_{axis}_a") for coord in coords for axis in ['x', 'y', 'z']]
        
        # 벡터화된 평균 계산
        results[:, 2*i] = np.nanmean(data[:, velocity_indices], axis=1)  # 평균 속도
        results[:, 2*i + 1] = np.nanmean(data[:, acceleration_indices], axis=1)  # 평균 가속도

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results, columns=[f"{name}_mean_v" for name in symptom_names] + [f"{name}_mean_a" for name in symptom_names])

    return results_df

