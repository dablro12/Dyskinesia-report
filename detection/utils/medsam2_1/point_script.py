import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# matplotlib 위젯 백엔드 사용 설정

def select_point(video_path):
    """
    주어진 비디오의 첫 프레임에서 한 지점을 선택하고 해당 좌표를 반환하는 함수입니다.

    Parameters:
    - video_path (str): 비디오 파일 경로

    Returns:
    - coords (tuple): 선택된 좌표 (x, y)
    """
    # 비디오 파일 존재 여부 확인
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return None

    # 비디오의 첫 프레임 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open the video file: {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Failed to read the video frame.")
        return None

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 좌표 저장을 위한 리스트 초기화
    coords = []

    # 마우스 클릭 이벤트 핸들러 함수 정의
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:
            print(f'x = {ix}, y = {iy}')
            coords.append((ix, iy))
            # 선택한 위치에 빨간 점 표시
            ax.plot(ix, iy, 'ro')
            fig.canvas.draw()
            # 이벤트 연결 해제 (한 번 클릭 후 종료)
            fig.canvas.mpl_disconnect(cid)
            print('좌표 선택이 완료되었습니다.')
            print('선택된 좌표:', list(coords[0]))
        else:
            print("Clicked outside the axes bounds.")

    # 그림과 축 생성
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('Click on the image to select a point')

    # 이벤트 연결
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # 좌표 선택이 완료될 때까지 대기
    plt.show()
    
    # 좌표 반환
    if coords:
        return coords[0]
    else:
        print('좌표가 선택되지 않았습니다.')
        return None
