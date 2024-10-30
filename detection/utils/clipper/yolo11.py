from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import torch
from collections import deque
import shutil

class yolo_Clipper:
    def __init__(self, ckpt_path: str, class_id: int = 0, tracker_type: str = 'CSRT', padding: int = 50, display: bool = False):
        """
        YOLO 모델을 초기화하고 추적기 설정을 구성합니다.

        :param ckpt_path: YOLO 모델 파일 경로 (예: "yolov8n.pt")
        :param class_id: YOLO 모델에서 사람 클래스의 ID (COCO 데이터셋 기준 일반적으로 0)
        :param tracker_type: 사용할 OpenCV 추적기 유형 ('CSRT', 'KCF', 'MOSSE', 'GOTURN', 'BOOSTING' 등)
        :param padding: 바운딩 박스 주위에 추가할 패딩 픽셀 수
        :param display: 추적 중 프레임을 시각화할지 여부
        """
        self.model = self.YOLO_model_loader(ckpt_path)
        self.class_id = class_id  # COCO 데이터셋에서 사람 클래스 ID는 일반적으로 0
        self.tracker_type = tracker_type
        self.padding = padding
        self.display = display  # 프레임 시각화 여부
        self.frame_buffer = deque(maxlen=10)  # 최근 10 프레임 저장

    def YOLO_model_loader(self, ckpt_path: str):
        """
        YOLO 모델을 로드합니다.

        :param ckpt_path: YOLO 모델 파일 경로
        :return: 로드된 YOLO 모델
        """
        try:
            start_time = time.time()
            # GPU 사용 가능 여부 확인
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            YOLO_Estim_model = YOLO(ckpt_path).to(device)
            load_time = time.time() - start_time
            print(f"## [0-1] YOLO 모델이 성공적으로 로드되었습니다: {ckpt_path} (소요 시간: {load_time:.2f}초, 장치: {device})")
            return YOLO_Estim_model
        except Exception as e:
            print(f"## [0-Error!!] YOLO 모델 로드 중 오류 발생: {e}")
            raise

    def initialize_tracker(self, frame, bbox):
        """
        첫 번째 바운딩 박스를 사용하여 OpenCV 추적기를 초기화합니다.

        :param frame: 초기 비디오 프레임
        :param bbox: 바운딩 박스 튜플 (x, y, w, h)
        :return: 초기화된 추적기
        """
        try:
            # OpenCV 버전에 따라 추적기 생성 방법 다르게 처리
            if int(cv2.__version__.split('.')[0]) >= 4:
                if self.tracker_type == 'MOSSE':
                    tracker = cv2.legacy.TrackerMOSSE_create()
                elif self.tracker_type == 'KCF':
                    tracker = cv2.legacy.TrackerKCF_create()
                elif self.tracker_type == 'CSRT':
                    tracker = cv2.legacy.TrackerCSRT_create()
                elif self.tracker_type == 'GOTURN':
                    # GOTURN 모델 파일 경로 설정
                    goturn_prototxt_src = 'checkpoint/GOTURN/goturn.prototxt'
                    goturn_caffemodel_src = 'checkpoint/GOTURN/goturn.caffemodel'
                    goturn_prototxt_dst = 'goturn.prototxt'
                    goturn_caffemodel_dst = 'goturn.caffemodel'

                    # 현재 작업 디렉토리에 모델 파일 복사
                    shutil.copy(goturn_prototxt_src, goturn_prototxt_dst)
                    shutil.copy(goturn_caffemodel_src, goturn_caffemodel_dst)
                    
                    # 모델 파일 존재 여부 확인
                    if not (os.path.exists(goturn_prototxt_src) and os.path.exists(goturn_caffemodel_src)):
                        raise FileNotFoundError("GOTURN 모델 파일인 'goturn.prototxt'와 'goturn.caffemodel'을 'checkpoint/GOTURN/' 디렉토리에 배치하세요.")
                                        
                    # GOTURN 추적기 생성
                    tracker = cv2.TrackerGOTURN_create()
                    
                elif self.tracker_type == 'BOOSTING':
                    tracker = cv2.legacy.TrackerBoosting_create()
                else:
                    raise ValueError(f"지원하지 않는 추적기 유형: {self.tracker_type}")
            else:
                if self.tracker_type == 'MOSSE':
                    tracker = cv2.TrackerMOSSE_create()
                elif self.tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                elif self.tracker_type == 'CSRT':
                    tracker = cv2.TrackerCSRT_create()
                elif self.tracker_type == 'GOTURN':
                    # GOTURN 모델 파일 경로 설정
                    goturn_prototxt_src = 'checkpoint/GOTURN/goturn.prototxt'
                    goturn_caffemodel_src = 'checkpoint/GOTURN/goturn.caffemodel'
                    goturn_prototxt_dst = 'goturn.prototxt'
                    goturn_caffemodel_dst = 'goturn.caffemodel'
                    
                    # 현재 작업 디렉토리에 모델 파일 복사
                    shutil.copy(goturn_prototxt_src, goturn_prototxt_dst)
                    shutil.copy(goturn_caffemodel_src, goturn_caffemodel_dst)
                                        
                    # 모델 파일 존재 여부 확인
                    if not (os.path.exists(goturn_prototxt_src) and os.path.exists(goturn_caffemodel_src)):
                        raise FileNotFoundError("GOTURN 모델 파일인 'goturn.prototxt'와 'goturn.caffemodel'을 'checkpoint/GOTURN/' 디렉토리에 배치하세요.")
                    
                    # GOTURN 추적기 생성
                    tracker = cv2.TrackerGOTURN_create()
                    
                elif self.tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                else:
                    raise ValueError(f"지원하지 않는 추적기 유형: {self.tracker_type}")

            # 추적기 초기화
            tracker.init(frame, bbox)
            print(f"## [1-3] {self.tracker_type} 추적기가 초기화되었습니다.")

            if self.display:
                # 초기 프레임에 바운딩 박스 그리기 (디버깅용)
                frame_copy = frame.copy()
                x, y, w, h = bbox
                cv2.rectangle(frame_copy, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.imshow("Tracker Initialization", frame_copy)
                cv2.waitKey(1)  # 1ms 대기 후 창 닫기
                cv2.destroyWindow("Tracker Initialization")

            return tracker
        except AttributeError as e:
            print(f"## [1-Error!!] 추적기 초기화 중 오류 발생: {e}")
            raise
        except FileNotFoundError as e:
            print(f"## [1-Error!!] 추적기 초기화 중 오류 발생: {e}")
            raise
        except Exception as e:
            print(f"## [1-Error!!] 추적기 초기화 중 오류 발생: {e}")
            raise

    def display_first_frame(self, frame, boxes):
        """
        첫 번째 프레임에 감지된 사람들을 표시합니다.

        :param frame: 첫 번째 비디오 프레임
        :param boxes: 바운딩 박스 리스트
        """
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for idx, (x, y, w, h) in enumerate(boxes):
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 10, f'ID: {idx}', color='g', fontsize=12, backgroundcolor='white')
        plt.title("[First Frame] Select the ID of the person to track")
        plt.axis('off')
        plt.show()

    def calculate_iou(self, boxA, boxB):
        """
        두 바운딩 박스 간의 Intersection over Union (IoU)를 계산합니다.

        :param boxA: 첫 번째 바운딩 박스 (x, y, w, h)
        :param boxB: 두 번째 바운딩 박스 (x, y, w, h)
        :return: IoU 값
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        unionArea = boxAArea + boxBArea - interArea

        if unionArea == 0:
            return 0

        iou = interArea / unionArea
        return iou

    def predict_and_clip(self, video_path: str, save_dir: str = "clipped_output", output_filename: str = "output.mp4"):
        """
        입력 비디오에서 사람을 감지하고 특정 사람을 추적하여 클리핑된 비디오를 저장합니다.

        :param video_path: 입력 비디오 파일 경로
        :param save_dir: 클리핑된 비디오를 저장할 디렉토리
        :param output_filename: 저장할 클리핑된 비디오 파일 이름
        :return: 저장된 클리핑 비디오의 경로
        """
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"## [0-Error!!] 비디오 파일을 열 수 없습니다: {video_path}")
        else:
            print("## [0-2] 비디오 파일이 성공적으로 열렸습니다.")

        # 비디오 속성 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v'는 mp4 컨테이너와 호환됨

        # 첫 번째 프레임 읽기
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("## [0-Error!!] 비디오의 첫 번째 프레임을 읽을 수 없습니다.")

        # 첫 번째 프레임에서 사람 감지
        start_time = time.time()
        results = self.model(first_frame, verbose=False)
        detection_time = time.time() - start_time
        print(f"## [1-1] 사람 감지 완료 (소요 시간: {detection_time:.2f}초)")

        boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append((x1, y1, w, h))

        if not boxes:
            raise ValueError("## [1-Error!!] 첫 번째 프레임에서 사람이 감지되지 않았습니다.")
        else:
            print(f"## [1-2] 첫 번째 프레임에서 {len(boxes)}명의 사람이 감지되었습니다.")

        # 첫 번째 프레임에 감지된 모든 사람을 표시
        self.display_first_frame(first_frame, boxes)

        # 사용자로부터 추적할 사람의 ID 선택
        print("## [1-3] 추적할 사람의 ID를 입력하세요:")
        while True:
            try:
                selected_id = int(input(f"ID (0 ~ {len(boxes)-1}): "))
                if 0 <= selected_id < len(boxes):
                    break
                else:
                    print(f"## [1-Error!!] 유효한 ID를 입력하세요 (0 ~ {len(boxes)-1}).")
            except ValueError:
                print("## [1-Error!!] 정수를 입력하세요.")

        selected_bbox = boxes[selected_id]
        print(f"## [1-4] 선택된 바운딩 박스: {selected_bbox}")

        # 선택된 바운딩 박스를 기반으로 추적기 초기화
        tracker = self.initialize_tracker(first_frame, selected_bbox)

        # 패딩을 추가한 크기로 출력 비디오 설정
        x, y, w, h = selected_bbox
        x_padded = max(int(x - self.padding), 0)
        y_padded = max(int(y - self.padding), 0)
        x2_padded = min(int(x + w + self.padding), original_width)
        y2_padded = min(int(y + h + self.padding), original_height)
        cropped_width = x2_padded - x_padded
        cropped_height = y2_padded - y_padded

        # 비디오 라이터 초기화
        output_video_path = os.path.join(save_dir, output_filename)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (cropped_width, cropped_height))
        print(f"## [2-1] 출력 비디오 설정 완료: {cropped_width}x{cropped_height} 픽셀, 저장 경로: {output_video_path}")

        frame_idx = 1  # 첫 번째 프레임은 이미 읽었으므로 1부터 시작
        start_processing_time = time.time()
        max_reinitialize = 60  # 최대 재초기화 시도 횟수
        reinitialize_count = 0

        # 원래 추적 대상의 중심점 저장
        original_center = (x + w / 2, y + h / 2)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("## [2-2] 비디오 프레임 읽기 완료.")
                break

            # 추적기 업데이트
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)

                # 패딩 적용
                x_padded = max(x - self.padding, 0)
                y_padded = max(y - self.padding, 0)
                x2_padded = min(x + w + self.padding, original_width)
                y2_padded = min(y + h + self.padding, original_height)

                # 프레임 자르기
                cropped_frame = frame[y_padded:y2_padded, x_padded:x2_padded]

                # 출력 비디오에 자른 프레임 쓰기
                out.write(cropped_frame)

                if self.display:
                    # 현재 추적 중인 프레임에 바운딩 박스 그리기 (디버깅용)
                    cv2.rectangle(frame, (int(x_padded), int(y_padded)), (int(x2_padded), int(y2_padded)), (0, 255, 0), 2)
                    cv2.imshow("Tracking", frame)

                # 프레임 버퍼에 현재 프레임 저장
                self.frame_buffer.append((frame.copy(), bbox))

            else:
                print(f"## [2-Error!!] 프레임 {frame_idx}에서 추적 실패. 재탐지 시도.")
                reinitialize_count += 1
                if reinitialize_count > max_reinitialize:
                    print("## [2-Error!!] 최대 재초기화 시도 횟수를 초과했습니다. 추적을 중지합니다.")
                    break

                # 현재 프레임에서 재탐지
                start_time = time.time()
                results = self.model(frame, verbose=False)
                detection_time = time.time() - start_time
                print(f"## [2-Redetect] 재탐지 완료 (소요 시간: {detection_time:.2f}초)")

                new_boxes = []
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == self.class_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w_new = x2 - x1
                            h_new = y2 - y1
                            new_boxes.append((x1, y1, w_new, h_new))

                if not new_boxes:
                    print("## [2-Redetect-Error!!] 재탐지된 사람이 없습니다. 추적을 중지합니다.")
                    break

                # 새로운 박스들 중 원래 박스와 가장 유사한 박스를 찾기 (IoU 기준)
                best_iou = 0
                best_box = None
                for box in new_boxes:
                    iou = self.calculate_iou(selected_bbox, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_box = box

                # 유사한 박스가 일정 IoU 이상일 경우 재초기화
                if best_iou > 0.3 and best_box is not None:
                    print(f"## [2-Redetect-Info] 유사한 박스를 발견했습니다. IoU: {best_iou:.2f}. 재초기화합니다.")
                    tracker = self.initialize_tracker(frame, best_box)
                    selected_bbox = best_box  # 업데이트된 박스를 원래 박스로 설정
                else:
                    print("## [2-Redetect-Error!!] 유사한 박스를 발견하지 못했습니다. 추적을 중지합니다.")
                    break

                if self.display and best_box:
                    # 현재 추적 중인 프레임에 바운딩 박스 그리기 (디버깅용)
                    x_new, y_new, w_new, h_new = best_box
                    cv2.rectangle(frame, (int(x_new), int(y_new)), (int(x_new + w_new), int(y_new + h_new)), (0, 0, 255), 2)
                    cv2.imshow("Tracking", frame)

                # 프레임 버퍼에 현재 프레임 저장
                self.frame_buffer.append((frame.copy(), best_box if best_box else None))

            # 프로그레스 표시 (매 100프레임마다)
            if frame_idx % 100 == 0:
                elapsed_time = time.time() - start_processing_time
                print(f"## [2-{frame_idx}] {frame_idx} 프레임 처리 완료, 경과 시간: {elapsed_time:.2f}초")

            # 키 입력 대기 (디버깅용)
            if self.display and cv2.waitKey(1) & 0xFF == ord('q'):
                print("## [2-Info] 사용자가 추적을 중지했습니다.")
                break

            frame_idx += 1

        # 리소스 해제
        cap.release()
        out.release()
        if self.display:
            cv2.destroyAllWindows()
        total_time = time.time() - start_processing_time
        print(f"## [3] 클리핑 완료. 클리핑된 비디오는 '{output_video_path}'에 저장되었습니다.")
        print(f"## [3-1] 총 처리 시간: {total_time:.2f}초")

        return output_video_path

# 사용 예제
if __name__ == "__main__":
    # YOLO 클리퍼 초기화
    # "yolov8n.pt" 모델이 작업 디렉토리에 없으면 자동으로 다운로드됩니다.
    yolo_clipper = yolo_Clipper(
        ckpt_path="checkpoint/yolo/yolo11l.pt",
        class_id=0,
        tracker_type='BOOSTING',  # 'CSRT', 'KCF', 'MOSSE', 'BOOSTING', 'GOTURN' 중 하나 선택 가능
        padding=15,  # 패딩 픽셀 수 조정 가능
        display=False  # 프레임 시각화 여부 (디버깅 시 True로 설정)
    )

    # 입력 비디오 경로와 출력 디렉토리 설정
    input_video = "data/video/sample_2_2_1.mp4"
    output_directory = "data/clipped_video"

    # 예측 및 클리핑 수행
    try:
        clipped_video_path = yolo_clipper.predict_and_clip(
            video_path=input_video,
            save_dir=output_directory,
            output_filename=os.path.basename(input_video)  # 입력 비디오 파일 이름 사용
        )
        print(f"클리핑된 비디오는 다음 경로에 저장되었습니다: {clipped_video_path}")
    except Exception as e:
        print(f"클리핑 수행 중 오류 발생: {e}")
