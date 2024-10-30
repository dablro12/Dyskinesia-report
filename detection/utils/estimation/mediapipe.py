import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

## >>>>> PARAMETER SETTINGS <<<<<<
DETECTION_CONF = 0.5
PRESENCE_CONF = 0.5
TRACKING_CONF = 0.5
SEGMASK = False

class mp_PoseEstimation:
    """
        (Ref) : https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python?hl=ko#video_2
        (Class) mp_PoseEstimation : mediapipe PoseLandmarker를 이용한 자세 추정 클래스
        (Usage)
        1. init(model_path : str) 설정
        2. run(video_path : str) 설정 -> return : annot_imgs : list : 자세 추정 결과 프레임별 Annotation 된 리스트
        3. save(frames:np.array, save_path:str='result/test.mp4') 설정
    """
    def __init__(self, model_path='checkpoint/mediapipe/pose_landmarker_heavy.task'):
        self._init_mp(model_path)
    
    def _init_mp(self, model_path):
        """ _init_mp : mediapipe PoseLandmarker 모델 초기화 """
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.options = self.PoseLandmarkerOptions(
            # base_options=self.BaseOptions(model_asset_path=model_path), # FOR CPU
            base_options=self.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.GPU), # FOR GPU
            running_mode=self.VisionRunningMode.VIDEO,
            num_poses=1,  # 감지할 수 있는 최대 포즈 수
            min_pose_detection_confidence=0.25,  # 자세 감지에 필요한 최소 신뢰도 점수
            min_pose_presence_confidence=0.25,  # 포즈 존재의 최소 신뢰도 점수
            min_tracking_confidence=0.25,  # 자세 추적의 최소 신뢰도 점수
            output_segmentation_masks=False  # 분화 마스크를 출력 여부
        )
        print(f"### [0] mediapipe PoseLandmarker init complete")
        
    def select_fourcc(self, file_extension: str):
        """
            (func) select_fourcc : 파일 확장자에 따라 fourcc 설정하는 함수
            (input) file_extension : str : 파일 확장자
            (return) fourcc : str : .확장자명
        """
        if file_extension.lower() == 'avi':
            fourcc = 'XVID'
        else:
            fourcc = 'mp4v'
        print(f"### [3-2] SELECT_FOURCC : {fourcc}")
        return fourcc

    def get_video_info(self, video):
        """
            (func) get_video_info : 비디오 정보 저장하는 함수
            (input) video : cv2.VideoCapture : 비디오 객체
            (return) video_info : dict : 비디오 정보 딕셔너리
        """
        video_info = {
            "frame_width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            "frame_rate": video.get(cv2.CAP_PROP_FPS),  # float 유지
            "frame_size": (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        }
        
        print(f"### [1-1] GET_VIDEO_INFO Complete")
        print(f"#### [Ruuning Time] : {video.get(cv2.CAP_PROP_POS_MSEC)}")
        
        return video_info

    def load_video(self, video_path: str):
        """
            func : load_video : 비디오 로드하여 frame list를 반환하는 함수
            input : video_path : str : 비디오 경로
            output : frames : list : 프레임 리스트 -> np.array : shape = (frame, height, width, channel)
        """
        video = cv2.VideoCapture(video_path)
        frame_info = self.get_video_info(video)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        print(f"### [1-2] LOAD_VIDEO Complete")
        return np.array(frames), frame_info

    def save_video(self, frames: np.array, save_path: str = 'result/test.mp4'):
        # video로 저장하기
        height, width, channel = frames[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*self.select_fourcc(save_path.split('.')[-1]))
        out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"### [3-1] SAVE Complete : {save_path}")

    def save_coord(self, pose_landmarks_frames, save_path):
        """
            (func) save_coord : mediapipe PoseLandmarker 결과를 프레임별로 coor를 모두 저장해 dataframe으로 변환하는 함수
            (input) 
                pose_landmarks_frames : list : mediapipe PoseLandmarker 결과 프레임별 리스트
                save_path : str : 저장할 csv 파일 경로
            (output) coords_df : pd.DataFrame : 저장된 csv 파일을 DataFrame으로 반환
        """
        frame_coords = {}
        
        for frame, pose_landmarks_list in enumerate(pose_landmarks_frames):
            frame_coords[frame] = []
            if pose_landmarks_list:
                for idx in range(len(pose_landmarks_list)):  # 수정된 부분
                    pose_landmarks = pose_landmarks_list[idx]
                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                        for landmark in pose_landmarks
                    ])
                    frame_coords[frame].append(pose_landmarks_proto)
            else:
                # 포즈가 감지되지 않은 프레임의 경우 빈 리스트 유지
                pass

        # 데이터프레임 생성
        data = []
        for frame, poses in frame_coords.items():
            frame_data = {'frame': frame}
            for pose_idx, pose in enumerate(poses):
                for landmark_idx, landmark in enumerate(pose.landmark):
                    frame_data[f"pose_{pose_idx}_landmark_{landmark_idx}_x"] = landmark.x
                    frame_data[f"pose_{pose_idx}_landmark_{landmark_idx}_y"] = landmark.y
                    frame_data[f"pose_{pose_idx}_landmark_{landmark_idx}_z"] = landmark.z
            data.append(frame_data)
        
        coords_df = pd.DataFrame(data)
        coords_df.to_csv(save_path, index=False)
        print(f"### [4] SAVE COORD Complete : {save_path}")
        return coords_df
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        if not pose_landmarks_list:
            return annotated_image  # 포즈가 감지되지 않은 경우 원본 이미지 반환

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                for landmark in pose_landmarks
            ])
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return annotated_image

    def run(self, video_path):
        """
            (Func) run : 비디오 경로를 받아서 자세 추정을 수행하는 함수
            (Input) video_path : str : 비디오 경로
            (Output) annot_imgs : list : 자세 추정 결과 이미지 리스트 
        """
        frames, frame_info = self.load_video(video_path)
        frame_rate = frame_info['frame_rate']
        frame_duration_us = int(1_000_000 / frame_rate)  # 마이크로초 단위로 프레임 간격 계산
        frame_timestamp_us = 0  # 초기 타임스탬프 (마이크로초)
        
        pose_landmarks_li, annot_imgs = [], []
        print(f"### [2] Pose Estimation Start using Mediapipe")
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            for frame in frames:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_us)
                
                annotated_image = self.draw_landmarks_on_image(frame, detection_result)
                
                pose_landmarks_li.append(detection_result.pose_landmarks)
                annot_imgs.append(annotated_image)
                
                frame_timestamp_us += frame_duration_us  # 다음 프레임 타임스탬프
        
        return annot_imgs, pose_landmarks_li

# EXAMPLE
if __name__ == "__main__":
    from datetime import datetime
    # SETTING
    MM_MODEL_PATH = 'checkpoint/mediapipe/pose_landmarker_heavy.task'

    # Initialize 
    start_time = datetime.now() 
    mp_pose = mp_PoseEstimation(model_path = MM_MODEL_PATH) 
    print(f" ### [Process 0] mediapipe PoseLandmarker init time : {datetime.now() - start_time}")

    # Run 꼭 video_path 인자 설정
    ## results : list : 자세 추정 결과 프레임별 Annotation 된 리스트
    results, pose_coords = mp_pose.run(video_path = 'data/temp/[orig]yoga.mov') 
    print(f" ### [Process 1] mediapipe PoseLandmarker Running time : {datetime.now() - start_time}")

    # Save Video
    mp_pose.save_video(results, save_path = 'data/temp/[pose]yoga.mov')
    # Save Coordinate
    coords_df = mp_pose.save_coord(pose_landmarks_frames = pose_coords, save_path = 'data/temp/[pose]yoga.cov')
