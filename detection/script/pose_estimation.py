#! /usr/bin/env python
import sys, os 
sys.path.append('../')

# %% Mediapipe Pose Estimation using Mask Video
from utils.estimation.mediapipe import mp_PoseEstimation
from datetime import datetime
from script import df_preprocess


# %% Runner 
def pose_est_runner(MM_MODEL_PATH, INPUT_DATA_PATH):
    start_time = datetime.now()
    print(f" ### [Process 0] mediapipe PoseLandmarker init time : {datetime.now() - start_time}")
    mp_pose = mp_PoseEstimation(model_path= MM_MODEL_PATH)
    
    ## results : list : 자세 추정 결과 프레임별 Annotation 된 리스트
    results, pose_coords = mp_pose.run(video_path = INPUT_DATA_PATH) 
    print(f" ### [Process 1] mediapipe PoseLandmarker Running time : {datetime.now() - start_time}")
    # Save Video\
    SAVE_PATH = INPUT_DATA_PATH.replace(INPUT_DATA_PATH.split('/')[-2], 'PE_RESULT') # 파일이름은 동일하게 유지 폴더만 바꿈
    
    mp_pose.save_video(results, save_path = SAVE_PATH)
    
    # Save Coordinate
    coords_df = mp_pose.save_coord(pose_landmarks_frames = pose_coords, save_path = SAVE_PATH.replace('mp4', 'csv'))
    return SAVE_PATH.replace('mp4', 'csv')

# [Example] Usage
if __name__ == "__main__":
    # SETTING
    # Initialize 
    MM_MODEL_PATH = 'checkpoint/mediapipe/pose_landmarker_heavy.task', 
    USE_DATA_DIR = 'data/result/sam2_1_mask', 
    SAVE_DIR = 'data/result/pose_estimation'
    
    start_time = datetime.now() 
    mp_pose = mp_PoseEstimation(model_path = MM_MODEL_PATH) 
    print(f" ### [Process 0] mediapipe PoseLandmarker init time : {datetime.now() - start_time}")
    df = df_preprocess(csv_path = 'data/BACKUP/extract_video/select_video_label.csv')

    exist_files = os.listdir(SAVE_DIR)
    for row in df.iterrows():
        if row[1]['filename'] in exist_files:
            continue
        
        filename = row[1]['filename']
        video_path = os.path.join(USE_DATA_DIR, filename) # Input Video Path : Mask Video
        save_video_path = os.path.join(SAVE_DIR, filename) # Output Video Path : Pose Estimation Video
        save_coord_path = os.path.join(SAVE_DIR, filename.replace('mp4', 'csv')) # Output Video Path : Pose Estimation Video
        
        # Run 꼭 video_path 인자 확인 
        ## results : list : 자세 추정 결과 프레임별 Annotation 된 리스트
        results, pose_coords = mp_pose.run(video_path = video_path) 
        print(f" ### [Process 1] mediapipe PoseLandmarker Running time : {datetime.now() - start_time}")
        # Save Video
        mp_pose.save_video(results, save_path = save_video_path)
        # Save Coordinate
        coords_df = mp_pose.save_coord(pose_landmarks_frames = pose_coords, save_path = save_coord_path)

