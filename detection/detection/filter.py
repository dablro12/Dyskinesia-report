import pandas as pd 
import sys, os 
import json 
import pandas as pd
import numpy as np
from scipy import stats
from pykalman import KalmanFilter
# %% 이상치 처리 
def outlier_process(df, threshold = 3, interpolation_use = False, interpolation_method = 'linear'):
    clean_df = df.copy()
    numeric_cols = [col for col in clean_df.columns if col != 'frame']
    z_scores = np.abs(stats.zscore(clean_df[numeric_cols].dropna()))

    # 임계값 설정 (예: 3)
    outliers = (z_scores > threshold)

    # 이상치가 있는 프레임 마스킹
    outlier_frames = outliers.any(axis=1)
    print(f'이상치가 있는 프레임 수: {outlier_frames.sum()}')
    # 4. 이상치를 NaN으로 대체하고 보간

    # clean_df의 복사본 생성
    df_clean = clean_df.copy()

    # 이상치를 NaN으로 대체
    df_clean[numeric_cols] = df_clean[numeric_cols].mask(outliers, np.nan)
    if interpolation_use:
        # 선형 보간법 적용
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method= interpolation_method)
        # # 보간 후 남은 NaN을 이전/다음 값으로 채우기 (필요 시)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='bfill').fillna(method='ffill')
        return df_clean
    else:
        return df_clean
    
# %% Header 전처리 
def pos_coor_header_preprocessing(df):
    # frame 칼럼 지우기
    df = df.iloc[:, 1:]
    # header명 바꾸기 ex) pose_0_landmark_4_x -> 4_x
    df = df.rename(columns = {col: col.split('_')[-2] + '_' + col.split('_')[-1] for col in df.columns})

    # 1_x, 1_y, 1_z -> [1_x, 1_y, 1_z] 바꾸고 컬럼명은 1 으로 
    # First, extract the unique numbers from your column names
    numbers = sorted(set(col.split('_')[0] for col in df.columns))

    # For each number, create a new column with a list of [x, y, z] values
    for n in numbers:
        df[str(n)] = df[[f'{n}_x', f'{n}_y', f'{n}_z']].values.tolist()

    # Drop the original x, y, z columns
    df = df.drop(columns=[f'{n}_{axis}' for n in numbers for axis in ['x', 'y', 'z']])
    
    # column을 int로 바꿔서 정렬
    df.columns = df.columns.astype(int)
    df = df.sort_index(axis=1)
    
    return df

from scipy.ndimage import gaussian_filter1d
import pandas as pd

def gaussian_smoothing(df, sigma=1):
    """
    각 조인트의 [x, y, z] 좌표에 대해 가우시안 평활화를 적용
    Parameters:
    - df: DataFrame, 각 셀이 [x, y, z] 형태의 좌표를 담고 있는 데이터프레임
    - sigma: 가우시안 필터의 표준 편차, 스무딩 강도를 조절하는 파라미터
    """
    smoothed_df = df.copy()
    
    for col in df.columns:
        # 각 조인트의 x, y, z 좌표를 분리
        x_vals = [coord[0] for coord in df[col]]
        y_vals = [coord[1] for coord in df[col]]
        z_vals = [coord[2] for coord in df[col]]
        
        # 가우시안 평활화 적용
        x_smooth = gaussian_filter1d(x_vals, sigma=sigma).tolist()
        y_smooth = gaussian_filter1d(y_vals, sigma=sigma).tolist()
        z_smooth = gaussian_filter1d(z_vals, sigma=sigma).tolist()
        
        # [x, y, z] 형태로 다시 결합
        smoothed_df[col] = [[x_smooth[i], y_smooth[i], z_smooth[i]] for i in range(len(x_smooth))]
    
    return smoothed_df

def moving_average_smoothing(df, window_size=3):
    """
    각 조인트의 [x, y, z] 좌표에 대해 이동 평균을 적용
    Parameters:
    - df: DataFrame, 각 셀이 [x, y, z] 형태의 좌표를 담고 있는 데이터프레임
    - window_size: 이동 평균을 적용할 윈도우 크기
    """
    smoothed_df = df.copy()
    
    for col in df.columns:
        # 각 조인트의 x, y, z 좌표를 분리
        x_vals = [coord[0] for coord in df[col]]
        y_vals = [coord[1] for coord in df[col]]
        z_vals = [coord[2] for coord in df[col]]
        
        # 이동 평균 적용
        x_smooth = pd.Series(x_vals).rolling(window=window_size, min_periods=1).mean().tolist()
        y_smooth = pd.Series(y_vals).rolling(window=window_size, min_periods=1).mean().tolist()
        z_smooth = pd.Series(z_vals).rolling(window=window_size, min_periods=1).mean().tolist()
        
        # [x, y, z] 형태로 다시 결합
        smoothed_df[col] = [[x_smooth[i], y_smooth[i], z_smooth[i]] for i in range(len(x_smooth))]
    return smoothed_df


## %% Calculate Velocity and Acceleration
def calculate_velocity_acceleration(df, fps=30):
    # 좌표 데이터가 리스트 형태이므로, 이를 분리하여 각 축에 대해 별도의 컬럼으로 나눕니다.
    expanded_df = pd.DataFrame()

    # 각 joint 컬럼을 분리하여 새로운 데이터프레임에 추가
    for col in df.columns:
        expanded_df[[f'{col}_x', f'{col}_y', f'{col}_z']] = pd.DataFrame(df[col].tolist(), index=df.index)

    # 속도 계산 (1차 미분)
    velocity_df = expanded_df.diff() * fps

    # 가속도 계산 (2차 미분)
    acceleration_df = velocity_df.diff() * fps

    # 속도 및 가속도 컬럼 이름 설정
    velocity_df.columns = [f'{col}' for col in velocity_df.columns]
    acceleration_df.columns = [f'{col}' for col in acceleration_df.columns]

    # 599 frame 및 598 frame 에서 600 frame으로 변경 *앞쪽에 추가해주기
    velocity_df = velocity_df.shift(-1)
    acceleration_df = acceleration_df.shift(-2)
    
    # NaN 값 제거하지 말고 이전 값으로 채우기 
    velocity_df = velocity_df.fillna(method='bfill').fillna(method='ffill')
    acceleration_df = acceleration_df.fillna(method='bfill').fillna(method='ffill')

    # print(f'NaN values in velocity data: {velocity_df.isnull().sum().sum()}')
    # print(f'NaN values in velocity data: {acceleration_df.isnull().sum().sum()}')
    return velocity_df, acceleration_df

def fourier_transform_joint(df):
    """
    관절 좌표 데이터에 Fourier 변환을 적용하여 주파수 성분 분석
    Parameters:
    - df: DataFrame, 각 조인트의 [x, y, z] 좌표를 담고 있는 데이터프레임
    Returns:
    - transformed_df: DataFrame, Fourier 변환된 주파수 성분 데이터
    """
    velo = df.copy()
    freq_df = pd.DataFrame()
    for col in velo.columns:
        # 리스트로 만들기
        vals = velo[col].tolist()
        
        fft = np.fft.fft(vals)
        freq_df[col] = fft

    return freq_df


# %% Kalman Filter 적용
from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def standardize(data):
    """
    데이터 표준화 (평균 0, 표준편차 1)
    
    Parameters:
    - data: list or array, 원본 데이터
    
    Returns:
    - standardized_data: list, 표준화된 데이터
    - mean: float, 원본 데이터의 평균
    - std: float, 원본 데이터의 표준편차
    """
    mean = np.mean(data)
    std = np.std(data)
    standardized_data = [(x - mean) / std for x in data]
    return standardized_data, mean, std

def rescale(data, mean, std):
    """
    표준화된 데이터를 원래 스케일로 복원
    
    Parameters:
    - data: list or array, 표준화된 데이터
    - mean: float, 원본 데이터의 평균
    - std: float, 원본 데이터의 표준편차
    
    Returns:
    - rescaled_data: list, 원래 스케일로 복원된 데이터
    """
    rescaled_data = [x * std + mean for x in data]
    return rescaled_data

def apply_kalman_filter_with_standardization(df):
    """
    각 조인트의 [x, y, z] 좌표에 표준화 후 칼만 필터를 적용하고 다시 스케일링합니다.
    
    Parameters:
    - df: DataFrame, 각 셀이 [x, y, z] 형태의 좌표를 담고 있는 데이터프레임
    
    Returns:
    - filtered_df: DataFrame, 칼만 필터가 적용된 데이터프레임
    """
    filtered_df = df.copy()
    
    for col in df.columns:
        # 각 조인트의 x, y, z 좌표를 분리
        x_vals = [coord[0] for coord in df[col]]
        y_vals = [coord[1] for coord in df[col]]
        z_vals = [coord[2] for coord in df[col]]
        
        # 표준화
        x_vals_std, x_mean, x_std = standardize(x_vals)
        y_vals_std, y_mean, y_std = standardize(y_vals)
        z_vals_std, z_mean, z_std = standardize(z_vals)
        
        # 칼만 필터 객체 생성
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)  # N(0, 1)로 초기화 및 관측 모델은 관측값 그대로 사용
        
        # 표준화된 데이터에 칼만 필터 적용
        x_filtered_std = kf.smooth(np.array(x_vals_std).reshape(-1, 1))[0].flatten().tolist()
        y_filtered_std = kf.smooth(np.array(y_vals_std).reshape(-1, 1))[0].flatten().tolist()
        z_filtered_std = kf.smooth(np.array(z_vals_std).reshape(-1, 1))[0].flatten().tolist()
        
        # 필터링된 데이터를 원래 스케일로 복원
        x_filtered = rescale(x_filtered_std, x_mean, x_std)
        y_filtered = rescale(y_filtered_std, y_mean, y_std)
        z_filtered = rescale(z_filtered_std, z_mean, z_std)
        
        # [x, y, z] 형태로 다시 결합
        filtered_df[col] = [[x_filtered[i], y_filtered[i], z_filtered[i]] for i in range(len(x_filtered))]
    
    # NaN 값 처리 : 앞쪽 값으로 채우기
    filtered_df = filtered_df.fillna(method='ffill')
    # NaN 값 처리 : 뒤쪽 값으로 채우기
    filtered_df = filtered_df.fillna(method='bfill')
    
    return filtered_df

# %% Paricle Filter 적용
import numpy as np
import pandas as pd
from numpy.random import randn
from filterpy.monte_carlo import systematic_resample

class ParticleFilter:
    def __init__(self, num_particles, state_dim, initial_state=None):
        """
        입자 필터 초기화
        
        Parameters:
        - num_particles: int, 사용할 입자의 수
        - state_dim: int, 상태 벡터의 차원 (예: 3D 좌표이면 3)
        - initial_state: array-like, 초기 상태 (기본값: 무작위)
        """
        self.num_particles = num_particles  # 입자의 수
        self.state_dim = state_dim          # 상태의 차원
        self.particles = np.random.randn(num_particles, state_dim)  # 초기 입자들
        self.weights = np.ones(num_particles) / num_particles       # 초기 가중치
        
        if initial_state is not None:
            # 초기 상태가 주어졌다면, 입자 초기화를 이 상태 주변에서 수행
            self.particles += initial_state

    def predict(self, u=0, std=0.1):
        """
        상태 예측 단계
        
        Parameters:
        - u: array-like, 제어 입력 (기본값: 0)
        - std: float, 예측 노이즈의 표준편차 (기본값: 0.1)
        """
        noise = randn(self.num_particles, self.state_dim) * std
        self.particles += u + noise

    def update(self, z, z_std=0.1):
        """
        가중치 업데이트 단계
        
        Parameters:
        - z: array-like, 관측된 값
        - z_std: float, 관측 노이즈의 표준편차
        """
        # 입자와 관측치 간의 유클리드 거리 계산
        distance = np.linalg.norm(self.particles - z, axis=1)
        self.weights *= np.exp(-0.5 * (distance / z_std) ** 2)
        
        # 수치적 안전성을 위해 작은 값 추가 후 정규화
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        입자 재샘플링 단계
        """
        indexes = systematic_resample(self.weights)
        self.particles = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        상태 추정 단계
        
        Returns:
        - estimate: array-like, 추정된 상태
        """
        return np.average(self.particles, weights=self.weights, axis=0)

def apply_particle_filter(df, num_particles=1000, std=0.1, measurement_noise=0.1):
    """
    각 조인트의 [x, y, z] 좌표에 입자 필터를 적용합니다.
    
    Parameters:
    - df: DataFrame, 각 셀이 [x, y, z] 형태의 좌표를 담고 있는 데이터프레임
    - num_particles: int, 사용할 입자의 수 (기본값: 1000)
    - std: float, 예측 노이즈의 표준편차 (기본값: 0.1)
    - measurement_noise: float, 측정 노이즈의 표준편차 (기본값: 0.1)
    
    Returns:
    - filtered_df: DataFrame, 입자 필터가 적용된 데이터프레임
    """
    filtered_df = df.copy()

    for col in df.columns:
        # 각 조인트의 x, y, z 좌표를 분리
        x_vals = [coord[0] for coord in df[col]]
        y_vals = [coord[1] for coord in df[col]]
        z_vals = [coord[2] for coord in df[col]]
        
        # 입자 필터 객체 생성 (각 축에 대해 별도로 필터 적용)
        pf_x = ParticleFilter(num_particles=num_particles, state_dim=1, initial_state=[x_vals[0]])
        pf_y = ParticleFilter(num_particles=num_particles, state_dim=1, initial_state=[y_vals[0]])
        pf_z = ParticleFilter(num_particles=num_particles, state_dim=1, initial_state=[z_vals[0]])
        
        # 필터링된 데이터를 저장할 리스트
        x_filtered = []
        y_filtered = []
        z_filtered = []
        
        # 각 프레임에 대해 입자 필터 적용
        for x, y, z in zip(x_vals, y_vals, z_vals):
            # 예측 단계
            pf_x.predict(std=std)
            pf_y.predict(std=std)
            pf_z.predict(std=std)
            
            # 관측 업데이트 단계
            pf_x.update(z=np.array([x]), z_std=measurement_noise)
            pf_y.update(z=np.array([y]), z_std=measurement_noise)
            pf_z.update(z=np.array([z]), z_std=measurement_noise)
            
            # 재샘플링
            pf_x.resample()
            pf_y.resample()
            pf_z.resample()
            
            # 상태 추정
            x_filtered.append(pf_x.estimate()[0])
            y_filtered.append(pf_y.estimate()[0])
            z_filtered.append(pf_z.estimate()[0])
        
        # [x, y, z] 형태로 다시 결합
        filtered_df[col] = [[x_filtered[i], y_filtered[i], z_filtered[i]] for i in range(len(x_filtered))]
        
    # NaN 값 처리 : 앞쪽 값으로 채우기
    filtered_df = filtered_df.fillna(method='ffill')
    filtered_df = filtered_df.fillna(method='bfill')
    return filtered_df