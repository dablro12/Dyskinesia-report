import os, sys
sys.path.append('../../reference/sam2')
from sam2.build_sam import build_sam2_video_predictor
import cv2
import torch
import numpy as np 

def medsam2_1_inference(video_path, save_dir, checkpoint, model_cfg, coor):
    print(f"Processing video: {video_path}")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    # Initialize video reader and writer
    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    centerX, centerY = int(coor[0]), int(coor[1]) # Ex [755.0947580645161, 438.16456653225805]
    
    # Define the codec and create VideoWriter object
    output_path = os.path.join(save_dir, video_path.split('/')[-1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'mp4v' or others
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # 비디오 리더를 처음으로 리셋
    video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path)

        # Corrected: Use integers for frame_idx and obj_id
        frame_idx = 0  # Start frame index
        obj_id = 1     # Unique object identifier

        # Ensure points and labels are correctly formatted
        points = np.array([[centerX, centerY]])  # Shape: (1, 2)
        labels = np.array([1])                   # Shape: (1,)

        # Add new prompts and get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels
        )

        # Reset the video reader to the beginning
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare to iterate over frames and masks
        frame_num = 0

        # Iterate over the frames and masks
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            ret, frame = video_reader.read()
            if not ret:
                break  # End of video

            # Process the mask
            # masks shape: (num_objects, 1, height, width)
            # We assume num_objects = 1 for this example
            mask = masks[0, 0].cpu().numpy()  # Get the mask for the first object
            mask = (mask > 0.5).astype(np.uint8) * 255  # Binarize and scale to [0,255]
            
            # mask인 부분을 제외하고 검은색으로 처리
            overlay_frame = frame.copy()
            overlay_frame[mask == 0] = 0

            # Write the frame to the output video
            video_writer.write(overlay_frame)

            frame_num += 1

        # Release the video reader and writer
        video_reader.release()
        video_writer.release()