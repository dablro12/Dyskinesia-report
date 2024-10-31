#!/bin/bash
set -e
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# script/classifier.py 실행
PYTHON_SCRIPT="/home/eiden/eiden/pd-ai/script/classifier.py"

data_type='vel_acc_kalman_symptom'
# data_type='kalman_symptom'
data_dir='/home/eiden/eiden/pd-ai/data/detection_data'
fold_dir='/home/eiden/eiden/pd-ai/data/fold'
save_dir='/home/eiden/eiden/pd-ai/data/result/classifier/timesnet'
task_type="multi_classification"
LOG_FILE="$LOG_DIR/timesnet_${data_type}.txt"

# 현재 run.sh 경로에서 classifier.py 실행
nohup /home/eiden/miniconda3/envs/cv/bin/python "$PYTHON_SCRIPT" \
    --data_type $data_type \
    --data_dir $data_dir \
    --fold_dir $fold_dir \
    --task_type $task_type \
    --save_dir $save_dir > "$LOG_FILE" 2>&1 &
echo "Python script is running in the background. Logs are saved to $LOG_FILE"
# /home/eiden/miniconda3/envs/cv/bin/python "$PYTHON_SCRIPT" \
#     --data_type $data_type \
#     --data_dir $data_dir \
#     --fold_dir $fold_dir \
#     --save_dir $save_dir
