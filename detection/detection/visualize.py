import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

def visualize_roi_outlier_plot(roi, df):
    roi_coor = df[roi]

    # %% plotting 각 관절별 좌표 동향 파악 내부에는 [x, y, z] 형태로 저장되어 있음
    fig, ax = plt.subplots(figsize=(10, 6))

    for col in roi_coor.columns:
        # Extract the list of [x, y, z] for each frame
        coord_list = roi_coor[col].tolist()
        
        # Separate x, y, z components
        x_vals = [coord[0] for coord in coord_list]
        y_vals = [coord[1] for coord in coord_list]
        z_vals = [coord[2] for coord in coord_list]
        
        # Plot x, y, z components separately
        ax.plot(roi_coor.index, x_vals, label=f'Joint {col} - x')
        ax.plot(roi_coor.index, y_vals, label=f'Joint {col} - y')
        ax.plot(roi_coor.index, z_vals, label=f'Joint {col} - z')

    ax.set_title('Joint Coordinates Over Time')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Coordinate Value')
    # 우상단에 legend 표시
    ax.legend(loc='upper right')
    plt.show()

# %% Visualize the Velocity and Acceleration
def visualize_velocity_acceleration(velocity_df, acceleration_df, joint_id):
    # Select the joint of interest
    joint_id = 1
    joint_cols = [col for col in velocity_df.columns if col.startswith(str(joint_id))]

    # Plot the data
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the velocity
    ax[0].plot(velocity_df.index, velocity_df[joint_cols])
    ax[0].set_title(f'Velocity of Joint {joint_id}')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Velocity')

    # Plot the acceleration
    ax[1].plot(acceleration_df.index, acceleration_df[joint_cols])
    ax[1].set_title(f'Acceleration of Joint {joint_id}')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Acceleration')

    plt.tight_layout()
    plt.show()

# %% Visualize 3D Whole Body Animation
class Whole3DVisualizer:
    def __init__(self, df, output_path='whole3d_animation.mp4', fps=30):
        """
        3D 관절 시각화 및 애니메이션을 위한 클래스
        
        Parameters:
        - df: DataFrame, 관절 데이터 (각 행은 프레임을 나타내고, 열은 조인트별 [x, y, z] 좌표)
        - output_path: str, 출력할 비디오 파일 경로 (기본값: 'whole3d_animation.mp4')
        - fps: int, 초당 프레임 수 (기본값: 30)
        """
        self.df = df
        self.output_path = output_path
        self.fps = fps
        self.frames = self.prepare_data()  # 데이터를 미리 준비합니다.
        
        # 첨부된 이미지에 따른 조인트 간 연결 관계
        self.joint_connections = [
            # 머리
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # 얼굴과 귀
            # 왼쪽 팔
            (11, 13), (13, 15), (15, 17),
            # 오른쪽 팔
            (12, 14), (14, 16), (16, 18),
            # 몸통
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 왼쪽 다리
            (23, 25), (25, 27), (27, 29),
            # 오른쪽 다리
            (24, 26), (26, 28), (28, 30)
        ]

    def prepare_data(self):
        """
        데이터 준비: DataFrame에서 각 프레임의 모든 조인트 좌표를 추출하여 리스트로 반환
        """
        frames = []
        for _, row in self.df.iterrows():
            # 각 프레임의 모든 조인트 좌표를 수집 (각 열이 [x, y, z] 형태라고 가정)
            frame_coords = np.array([coord for coord in row])
            frames.append(frame_coords)
        return frames

    def get_axis_limits(self):
        """
        전체 데이터에 대해 x, y, z 좌표의 최소값과 최대값을 계산하여 축 범위 설정
        """
        all_coords = np.vstack(self.frames)  # 모든 프레임의 좌표를 하나의 배열로 결합
        x_min, y_min, z_min = all_coords[:, 0].min(), all_coords[:, 1].min(), all_coords[:, 2].min()
        x_max, y_max, z_max = all_coords[:, 0].max(), all_coords[:, 1].max(), all_coords[:, 2].max()

        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    def update_graph(self, frame_data, graph, lines):
        """
        그래프 업데이트 함수 (애니메이션에서 각 프레임마다 호출)
        """
        # 스캐터 플롯 업데이트 (조인트 위치)
        graph._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        
        # 지정된 조인트 간 선 연결
        for i, (start, end) in enumerate(self.joint_connections):
            x_line = [frame_data[start, 0], frame_data[end, 0]]
            y_line = [frame_data[start, 1], frame_data[end, 1]]
            z_line = [frame_data[start, 2], frame_data[end, 2]]
            lines[i].set_data(x_line, y_line)
            lines[i].set_3d_properties(z_line)

        return graph, lines

    def visualize(self):
        """
        3D 시각화를 통해 비디오 애니메이션 생성
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 축 범위 설정 (데이터 범위에 맞게 자동으로 조정)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.get_axis_limits()
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # 카메라 시점 설정 (잘 보이도록)
        ax.view_init(elev=20, azim=30)

        # 조인트 스캐터 플롯 초기화
        graph = ax.scatter([], [], [], c='b', marker='o', s=100)  # s는 점의 크기

        # 선 초기화 (조인트 간 연결선 생성)
        lines = [ax.plot([], [], [], c='r')[0] for _ in range(len(self.joint_connections))]

        # 애니메이션 생성
        ani = FuncAnimation(fig, self.update_graph, frames=self.frames, fargs=(graph, lines), interval=1000/self.fps)

        # 비디오로 저장 (ffmpeg 필요)
        ani.save(self.output_path, writer='ffmpeg', fps=self.fps)
        print(f"Animation saved to {self.output_path}")

# %% Visualize 3D Joint Animation
class Joint3DVisualizer:
    def __init__(self, df, joint_indices, output_path='animation.mp4', fps=30):
        """
        3D 관절 시각화 및 애니메이션을 위한 클래스
        
        Parameters:
        - df: DataFrame, 관절 데이터 (각 행은 프레임을 나타내고, 열은 조인트별 [x, y, z] 좌표)
        - joint_indices: List[int], 시각화할 조인트 인덱스
        - output_path: str, 출력할 비디오 파일 경로 (기본값: 'animation.mp4')
        - fps: int, 초당 프레임 수 (기본값: 30)
        """
        self.df = df
        self.joint_indices = joint_indices
        self.output_path = output_path
        self.fps = fps
        self.frames = self.prepare_data()  # 데이터를 미리 준비합니다.
        self.axis_limits = self.get_axis_limits()  # 축 범위 설정

    def prepare_data(self):
        """
        데이터 준비: DataFrame에서 각 프레임의 조인트 좌표를 추출하여 리스트로 반환
        """
        frames = []
        for _, row in self.df.iterrows():
            # 각 조인트의 [x, y, z] 좌표를 가져옵니다.
            frame_coords = np.array([row[idx] for idx in self.joint_indices])
            frames.append(frame_coords)
        return frames

    def get_axis_limits(self):
        """
        전체 데이터에 대해 x, y, z 좌표의 최소값과 최대값을 계산하여 축 범위 설정
        """
        all_coords = np.vstack(self.frames)  # 모든 프레임의 좌표를 하나의 배열로 결합
        x_min, y_min, z_min = all_coords[:, 0].min(), all_coords[:, 1].min(), all_coords[:, 2].min()
        x_max, y_max, z_max = all_coords[:, 0].max(), all_coords[:, 1].max(), all_coords[:, 2].max()

        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    def update_graph(self, frame_data, graph, lines):
        """
        그래프 업데이트 함수 (애니메이션에서 각 프레임마다 호출)
        """
        # 스캐터 플롯 업데이트 (조인트 위치)
        graph._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])
        
        # 각 조인트를 선으로 모두 연결
        for i, line in enumerate(lines):
            for j in range(i + 1, len(frame_data)):
                x_line = [frame_data[i, 0], frame_data[j, 0]]
                y_line = [frame_data[i, 1], frame_data[j, 1]]
                z_line = [frame_data[i, 2], frame_data[j, 2]]
                line.set_data(x_line, y_line)
                line.set_3d_properties(z_line)

        return graph, lines

    def visualize(self):
        """
        3D 시각화를 통해 비디오 애니메이션 생성
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 축 범위 설정 (데이터 범위에 맞게 자동으로 조정)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.axis_limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # 카메라 시점 설정 (잘 보이도록)
        ax.view_init(elev=20, azim=30)

        # 조인트 스캐터 플롯 초기화
        graph = ax.scatter([], [], [], c='b', marker='o', s=100)  # s는 점의 크기

        # 조인트를 연결하는 선들 초기화
        lines = [ax.plot([], [], [], c='r')[0] for _ in range(len(self.joint_indices))]

        # 애니메이션 생성
        ani = FuncAnimation(fig, self.update_graph, frames=self.frames, fargs=(graph, lines), interval=1000/self.fps)

        # 비디오로 저장 (ffmpeg 필요)
        ani.save(self.output_path, writer='ffmpeg', fps=self.fps)
        print(f"Animation saved to {self.output_path}")


def plot_frequency_components(fft_data_x, fft_data_y, fft_data_z, sampling_rate, joint_name):
    """
    Fourier 변환된 주파수 성분을 시각화 (x, y, z 좌표를 한 번에 표시)
    Parameters:
    - fft_data_x: np.array, Fourier 변환된 x 좌표 데이터
    - fft_data_y: np.array, Fourier 변환된 y 좌표 데이터
    - fft_data_z: np.array, Fourier 변환된 z 좌표 데이터
    - sampling_rate: 샘플링 주파수 (초당 데이터 샘플 수)
    - joint_name: str, 시각화할 조인트 이름
    """
    n = len(fft_data_x)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)  # 주파수 계산

    # Magnitude 계산
    magnitude_x = np.abs(fft_data_x)
    magnitude_y = np.abs(fft_data_y)
    magnitude_z = np.abs(fft_data_z)

    # 주파수 성분 시각화
    plt.figure(figsize=(10, 6))

    # X 좌표 주파수 성분
    plt.plot(freq[:n//2], magnitude_x[:n//2], label=f'{joint_name} X')

    # Y 좌표 주파수 성분
    plt.plot(freq[:n//2], magnitude_y[:n//2], label=f'{joint_name} Y')

    # Z 좌표 주파수 성분
    plt.plot(freq[:n//2], magnitude_z[:n//2], label=f'{joint_name} Z')

    # 그래프 설정
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Components for {joint_name}')
    plt.legend()
    plt.show()
    
def plot_frequency_components_without_dc(fft_data_x, fft_data_y, fft_data_z, sampling_rate, joint_name):
    """
    Fourier 변환된 주파수 성분을 시각화 (x, y, z 좌표를 한 번에 표시) - DC 성분 제외
    Parameters:
    - fft_data_x: np.array, Fourier 변환된 x 좌표 데이터
    - fft_data_y: np.array, Fourier 변환된 y 좌표 데이터
    - fft_data_z: np.array, Fourier 변환된 z 좌표 데이터
    - sampling_rate: 샘플링 주파수 (초당 데이터 샘플 수)
    - joint_name: str, 시각화할 조인트 이름
    """
    n = len(fft_data_x)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)  # 주파수 계산

    # Magnitude 계산
    magnitude_x = np.abs(fft_data_x)
    magnitude_y = np.abs(fft_data_y)
    magnitude_z = np.abs(fft_data_z)

    # DC 성분 제외 (0Hz 성분 제외)
    plt.figure(figsize=(10, 6))
    plt.plot(freq[1:n//2], magnitude_x[1:n//2], label=f'{joint_name} X')  # X 좌표
    plt.plot(freq[1:n//2], magnitude_y[1:n//2], label=f'{joint_name} Y')  # Y 좌표
    plt.plot(freq[1:n//2], magnitude_z[1:n//2], label=f'{joint_name} Z')  # Z 좌표

    # 그래프 설정
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Components for {joint_name} (without DC)')
    plt.legend()
    plt.show()