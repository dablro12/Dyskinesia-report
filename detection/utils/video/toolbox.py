import sys, os 
import cv2 
import numpy as np 
from typing import *
import PIL.Image as Image
import matplotlib.pyplot as plt
from datetime import datetime

def time_calculator(time:int):
    """
    func : time_calculator : 시간을 받아서 시간, 분, 초로 변환하는 함수
    input : time : int : 시간
    output : hour : int : 시간
             minute : int : 분
             second : int : 초
    """
    hour = time // 3600
    minute = (time % 3600) // 60
    second = (time % 3600) % 60
    return hour, minute, second
    

def get_video_info(video):
    video_info = {
        "frame_width" : int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height" : int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count" : int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        "frame_rate" : int(video.get(cv2.CAP_PROP_FPS)),
        "frame_size" : (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    }
    return video_info

def load_video(video_path:str):
    """
        func : load_video : 비디오 로드하여 frame list를 반환하는 함수
        input : video_path : str : 비디오 경로
        output : frames : list : 프레임 리스트 -> np.array : shape = (frame, height, width, channel)
    """

    video = cv2.VideoCapture(video_path)
    frame_info = get_video_info(video)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return np.array(frames), frame_info

def select_fourcc(file_extension:str):
    if file_extension == 'avi':
        fourcc = 'XVID'
    else:
        fourcc = 'mp4v'
    return fourcc

def save_video(frames:np.array,save_path:str='result/test.mp4'):
    # video로 저장하기
    height, width, channel = frames[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*select_fourcc(save_path.split('.')[-1]))
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"save video : {save_path}")
    
def show_frame(frames:np.array, frame_index = 0):
    show_img = frames[frame_index]
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    plt.imshow(show_img ) #색상이 이상하게 나오는 이유는 opencv는 BGR로 읽어오기 때문이기에 RGB로 변환해주어야함
    plt.title(f"frame : {frame_index}")
    plt.show()

# %% Example
if __name__ == "__main__":
    video_path = 'data/1.mp4'
    frames, frame_info = load_video(video_path)
    print(frame_info)
    show_frame(frames, frame_index=0)
    save_video(frames, save_path='result/test.mp4')