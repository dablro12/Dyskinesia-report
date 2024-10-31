import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# %% [Pose Estimation] Module
from script.pose_estimation import pose_est_runner 
from detection.visualize import *
from detection.filter import *
from detection.preprocess import data_preprocessor, data_postprocessor, create_windows
from utils.final_csv import result_post_process
from script.infer import ml_classifier

# %% [Load Libraries]
import pandas as pd 
import argparse
import time
# %% [Pose Estimation] 전역변수
POSE_GUIDE = {"foot_left": [27, 29, 31], "foot_right": [28, 30, 32], "arm_left": [11, 13, 15], "arm_right": [12, 14, 16], "lower_body_right": [24, 26, 28], "lower_body_left": [23, 25, 27], "upper_body_right": [12, 24, 26], "upper_body_left": [11, 23, 25]}
WINDOW_SIZE = 600
# %% 입력 해줘야하는 변수 -> argparser 에서 설정해주세요
INPUT_DATA_PATH = 'PE_INPUT/1-1_1.mp4'
GUIDE_BOOK_PATH = "configs/mediapipe_pose_guide.json"

# CKPT Path 도 설정해주셔도 되고, 미리 넣어놨어요.
MM_MODEL_PATH = 'config/weight/pose_landmarker_heavy.task'
DETECT_MODEL_PATH = "config/weight/mult_label_classifier.pth"

# %% [Example] Usage
if __name__ == "__main__": 
    before = time.time()
    # [0] parser 설정 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str, help='input data path : ex) PE_INPUT/1-1_1.mp4')
    parser.add_argument('--guide_book_path', type=str, help='guide book path : ex) configs/mediapipe_pose_guide.json')
    parser.add_argument('--mm_model_path', type=str, help='mediapipe model path : ex) config/weight/pose_landmarker_heavy.task')
    parser.add_argument('--detect_model_path', type=str, help='detection model path : ex) config/weight/mult_label_classifier.pth')
    args = parser.parse_args()
    # [1~4] Mediapipe : Coordinate Estimation
    coords_df_path = pose_est_runner(MM_MODEL_PATH = args.mm_model_path, INPUT_DATA_PATH = args.input_data_path)
    # [5] Data Preprocessing using Kalman Filter & Outlier Removal using Z-Score
    preprocessor = data_preprocessor(
            example_csv_path = coords_df_path,
            guide_book_path = args.guide_book_path,
            visualize_use= False, # 시각화 사용 여부
            visualize_roi_name= 'Foot_Left', # 시각화할 관절 이름
            visualize_3d_use = False, # 3D 시각화 사용 여부
            visualize_save_dir = 'visualization') # 시각화 저장 디렉토리
    _, vel_df, acc_df, _, _ = preprocessor.run()
    
    # [6] Data Postprocessing for ROI Selection of POSE_GUIDE Point
    vel_acc_df_save_path = data_postprocessor(vel_df, acc_df, coords_df_path, POSE_GUIDE)
    vcc_acc_combined_df = pd.read_csv(vel_acc_df_save_path)
    # [7] Create Windows Example 1200개 frame -> 600개 window size로 순차 분할
    windows = create_windows(vcc_acc_combined_df, window_size=WINDOW_SIZE, stride = 1)
    # [8] Dyskinesia Detection using TimesNet
    pred = ml_classifier(
        ckpt_path = args.detect_model_path,
        data_path = vel_acc_df_save_path, # vel
        windows = windows
    )
    print(f"Prediction : {pred.shape}")
    # [9] Data Processing
    df = pd.read_csv(coords_df_path.replace('PE_RESULT', 'PE_RESULT'))
    final_df = result_post_process(coor_df = df, vcc_acc_combined_df=vcc_acc_combined_df, pred = pred, POSE_GUIDE = POSE_GUIDE, window_size=WINDOW_SIZE)
    
    final_df.to_csv(coords_df_path.replace('PE_RESULT', 'PE_RESULT'), index=False)
    print(f"Total Time : {time.time() - before}")
    