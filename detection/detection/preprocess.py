from detection.filter import *
from detection.visualize import *

def data_postprocessor(vel_df, acc_df, coords_df_path, POSE_GUIDE):
    # acc_df의 컬럼명에 _acc 추가
    vel_df.columns = [f'{col}_v' for col in vel_df.columns]
    acc_df.columns = [f'{col}_a' for col in acc_df.columns]
    
    vel_acc_df = pd.concat([vel_df, acc_df], axis=1)
    vel_acc_df_save_path = coords_df_path.replace('.csv', '_vel_acc.csv')
    # column에서 POSE GUIDE에 있는 관절 숫자가 있는지 확인후 해당 관절만 가져오기 
    vel_acc_df = vel_acc_df[[col for col in vel_acc_df.columns if int(col.split('_')[0]) in POSE_GUIDE['foot_left'] + POSE_GUIDE['foot_right'] + POSE_GUIDE['arm_left'] + POSE_GUIDE['arm_right'] + POSE_GUIDE['lower_body_right'] + POSE_GUIDE['lower_body_left'] + POSE_GUIDE['upper_body_right'] + POSE_GUIDE['upper_body_left']]]
    vel_acc_df.to_csv(vel_acc_df_save_path, index=False)
    print("### [6] Data Postprocessing Complete : ", vel_acc_df_save_path)
    
    return vel_acc_df_save_path
    
def create_windows(df, window_size=600, stride=1):
    windows = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        windows.append(df.iloc[start:end])
    return windows

class data_preprocessor:
    def __init__(self, example_csv_path, guide_book_path, visualize_roi_name = 'Foot_Left', visualize_use = False, visualize_3d_use = False, visualize_save_dir = None):
        """
        Parameters:
        - example_csv_path: str, 예시 데이터 CSV 파일 경로
        - guide_book_path: str, 관절 가이드북 JSON 파일 경로
        - visualize_use: bool, 시각화 사용 여부 (기본값: False)
        - visualize_roi_name: str, 시각화할 관절 이름 (기본값: 'Foot_Left')
        - visualize_save_dir: str, 시각화 결과 저장 디렉토리 (기본값: None)
        
        Roi List:
        - Mouth
        - Left_Eye
        - Right_Eye
        - Nose
        - Left_ear
        - Right_ear
        - Arm_Left
        - Arm_Right
        - Wrist_Left
        - Wrist_Right
        - Upper_Body_Left
        - Upper_Body_Right
        - Lower_Body_Left
        - Lower_Body_Right
        - Foot_Right
        - Foot_Left
        
        Usage:
        preprocessor = data_preprocessor(
            example_csv_path = 'example.csv',
            guide_book_path = 'guide.json',
            visualize_use = True,
            visualize_roi_name = 'Foot_Left',
            visualize_save_dir = 'visualization'
        )
        preprocessor.run()
        preprocessor.save(save_dir = "your/save/dir")
        
        """
        self.example_csv_path = example_csv_path
        self.guide_book_path = guide_book_path
        self.roi_name = visualize_roi_name
        self.visualize_use = visualize_use
        self.visualize_3d_use= visualize_3d_use
        self.visualize_save_dir = visualize_save_dir
    def clean_df_outsocing(self, df:pd.DataFrame) -> pd.DataFrame:
        # df의 각 row는 [x,y,z]로 되어 있음 column은 0~32 으로 되어 있음
        # row에 있는 리스트를 분리해서 이를 0_x, 0_y, 0_z로 변경
        df = df.applymap(lambda x: x.replace('[', '').replace(']', '').replace(' ', '').split(','))
        df = df.applymap(lambda x: [float(i) for i in x])
        df = df.applymap(lambda x: x[0] if len(x) == 1 else x)
        return df

    # %% Data Preprocessing
    def preprocess_func(self, example_csv_path, guide_book_path, roi_name = 'Foot_Left', visualize_use = False, visualize_3d_use = False, visualize_save_dir = None):
        df = pd.read_csv(example_csv_path)
        # 1. 이상치 처리
        outlier_df = outlier_process(df, threshold= 1.5, interpolation_use = True, interpolation_method = 'pad') #이상치제거
        # 2. Header 전처리
        outlier_df = pos_coor_header_preprocessing(outlier_df) # frame 제거 및  Header 전처리 
        origin_df = pos_coor_header_preprocessing(df) # frame 제거 및  Header 전처리
        # 3. 스무딩 처리
        # clean_df = moving_average_smoothing(outlier_df, window_size=2) # 이동평균
        # clean_df = gaussian_smoothing(outlier_df, sigma=1) # 가우시안 스무딩
        clean_df = apply_kalman_filter_with_standardization(outlier_df)  # 칼만 필터 적용
        # clean_df = apply_particle_filter(outlier_df)  # 입자 필터 적용
        
        print("df Nan Count : ", clean_df.isnull().sum().sum())
        # df = apply_Extended_kalman_filter_with_standardization(df)  # 확장 칼만 필터 적용
        # 4. 속도 및 가속도 계산
        velocity_df, acceleration_df = calculate_velocity_acceleration(clean_df, fps=30)
        
        # 5. FFT -> 주파수 성분 분석
        velocity_df_F = fourier_transform_joint(velocity_df)
        acceleration_df_F = fourier_transform_joint(acceleration_df)
        # %% Visualization
        if visualize_use:
            with open(guide_book_path, 'r') as f:
                guide_book = json.load(f)
                roi = guide_book[roi_name]
            visualize_roi_outlier_plot(roi, origin_df) # roi : dict : 관절명 : [x, y, z] 형태의 리스트
            visualize_roi_outlier_plot(roi, outlier_df) # roi : dict : 관절명 : [x, y, z] 형태의 리스트
            visualize_roi_outlier_plot(roi, clean_df) # roi : dict : 관절명 : [x, y, z] 형태의 리스트
            visualize_velocity_acceleration(velocity_df, acceleration_df, joint_id = 0)

            # %% 3D Visualization
            if visualize_3d_use:
                # Joint 3D Animation
                joint_3D_visualizer = Joint3DVisualizer(
                    df = clean_df,
                    joint_indices= roi, 
                    output_path = os.path.join(visualize_save_dir, example_csv_path.split('/')[-1].replace('.csv', '_joint.mp4')),
                    fps = 30
                )
                joint_3D_visualizer.visualize()

                #  Whole Body 3D Animation
                whole_body_3D_visualizer = Whole3DVisualizer(
                    df = clean_df,
                    output_path = os.path.join(visualize_save_dir, example_csv_path.split('/')[-1].replace('csv', '_whole.mp4')),
                )
                whole_body_3D_visualizer.visualize()

            # %% Visualize Frequency Components (Fourier Transform)
            # sampling_rate = 30 #샘플링 주파수 설정
            # plot_frequency_components(transformed_df['0_x'], transformed_df['0_y'], transformed_df['0_z'], sampling_rate, joint_name='Joint 0')
            # plot_frequency_components_without_dc(transformed_df['0_x'], transformed_df['0_y'], transformed_df['0_z'], sampling_rate, joint_name='Joint 0')
        return clean_df, velocity_df, acceleration_df, velocity_df_F, acceleration_df_F
    def run(self):
        self.df, self.velocity_df, self.acceleration_df, self.velocity_df_F, self.acceleration_df_F = self.preprocess_func(
            self.example_csv_path,
            self.guide_book_path,
            self.roi_name,
            self.visualize_use, # 시각화 사용 여부
            self.visualize_3d_use, # 3D 시각화 여부 
            self.visualize_save_dir # 시각화 저장 디렉토리
        )

        print("### [5] Data Preprocessing Done")
        # print(f"Data Shape : {self.df.shape}")
        # print(f"Velocity Data Shape : {self.velocity_df.shape}")
        # print(f"Acceleration Data Shape : {self.acceleration_df.shape}")
        # print(f"Velocity Fourier Data Shape : {self.velocity_df_F.shape}")
        # print(f"Acceleration Fourier Data Shape : {self.acceleration_df_F.shape}")
        return self.df, self.velocity_df, self.acceleration_df, self.velocity_df_F, self.acceleration_df_F
    def save(self, save_dir):
        # Mkdir DIR
        Folder_name = self.example_csv_path.split('/')[-1].replace('.csv', '')
        save_dir = os.path.join(save_dir, Folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save Data
        self.df.to_csv(os.path.join(save_dir, 'preprocessed_data.csv'), index = False)
        self.velocity_df.to_csv(os.path.join(save_dir, 'velocity_data.csv'), index = False)
        self.acceleration_df.to_csv(os.path.join(save_dir, 'acceleration_data.csv'), index = False)
        self.velocity_df_F.to_csv(os.path.join(save_dir, 'velocity_F_data.csv'), index = False)
        self.acceleration_df_F.to_csv(os.path.join(save_dir, 'acceleration_F_data.csv'), index = False)
        print("######################### Data Saved #########################")
