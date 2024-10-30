import sys, os 
sys.path.append('../')

import df_preprocess 

# %% SAM2.1 Patient Masking 
import os 
from utils.medsam2_1.point_script import select_point
from utils.medsam2_1.inference import medsam2_1_inference


if __name__ == "__main__":
    # Initialize the predictor
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    save_dir = '/home/eiden/eiden/pd-ai/data/result/sam2_1_mask'

    df = df_preprocess(csv_path = '/home/eiden/eiden/pd-ai/data/BACKUP/extract_video/select_video_label.csv')
    exist_files = os.listdir(save_dir)
    for row in df.iterrows():
        if row[1]['filename'] in exist_files:
            continue
        point = row[1]['coordinate']
        filename = row[1]['filename']
        video_path = os.path.join('/home/eiden/eiden/pd-ai/data/BACKUP/extract_video/datacut_20s', filename)
        medsam2_1_inference(video_path, save_dir, checkpoint, model_cfg, point)