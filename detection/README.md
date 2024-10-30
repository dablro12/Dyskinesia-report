# [Dyskinesia Module]
# Module List
- [Dyskinesia Detection](#dyskinesia-detection)
- [Dyskinesia Report](#dyskinesia-report)
- [Dyskinesia Severity](#dyskinesia-severity)
- [Dyskinesia Type](#dyskinesia-type)
- [Dyskinesia Video](#dyskinesia-video)
- [Dyskinesia Video Report](#dyskinesia-video-report)
- [Dyskinesia Video Severity](#dyskinesia-video-severity)
- [Dyskinesia Video Type](#dyskinesia-video-type)

# Dyskniesia PE & Multi Label Classifier ENV Setup
OS : Ubuntu 20.04
GPU : Nvidia RTX 4090
Require RAM / VRAM : >8GB / >8GB
CUDA / CUDNN : 12.1 / 8.9.2
Require Disk : 10GB

## Installation
```bash
%cd detection
conda create -n dyskinesia python=3.8
conda activate dyskinesia
pip install -r config/setup/eiden.txt
```

# Dyskinesia Detection Using Mediapipe Pose Landmarker & Multi Label Classifier
## Usage 
```bash
%cd detection
python runner.py --input_data_path {input_data_path} \
    --guide_book_path="configs/mediapipe_pose_guide.json" \
    --mm_model_path="config/weight/pose_landmarker_heavy.task" \
    --detect_model_path="config/weight/mult_label_classifier.pth"
```
## Exmples
```bash
%cd detection
python runner.py --input_data_path="PE_INPUT/1-1_1.mp4" \
    --guide_book_path="configs/mediapipe_pose_guide.json"
    --mm_model_path="config/weight/pose_landmarker_heavy.task"
    --detect_model_path="config/weight/mult_label_classifier.pth"
``` 