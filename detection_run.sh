#!/bin/bash
set -e

# 세팅 파일
GUIDE_BOOK_PATH="configs/mediapipe_pose_guide.json"
MM_MODEL_PATH="config/weight/pose_landmarker_heavy.task"
DETECTION_MODEL_PATH="config/weight/mult_label_classifier.pth"

# 입력 비디오 경로
INPUT_DATA_PATH="PE_INPUT/example_long_video.mp4"

cd detection
DETECTION_SCRIPT="detection.py"

# 환경 맞춰서 실행
/home/eiden/miniconda3/envs/cv/bin/python "$DETECTION_SCRIPT" \
    --input_data_path $INPUT_DATA_PATH \
    --guide_book_path $GUIDE_BOOK_PATH \
    --mm_model_path $MM_MODEL_PATH \
    --detect_model_path $DETECTION_MODEL_PATH

echo "Python script is running in the background. Logs are saved to detection/PE_RESULT"
