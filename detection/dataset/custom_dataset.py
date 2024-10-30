import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

POSE_GUIDE = {"foot_left": [27, 29, 31], "foot_right": [28, 30, 32], "arm_left": [11, 13, 15], "arm_right": [12, 14, 16], "lower_body_right": [24, 26, 28], "lower_body_left": [23, 25, 27], "upper_body_right": [12, 24, 26], "upper_body_left": [11, 23, 25]}
INFER_POSE_GUIDE = {
    "Foot_Left": 1,
    "Foot_Right": 2,
    "Arm_Left": 3,
    "Arm_Right": 4,
    "Lower_Body_Right": 5,
    "Lower_Body_Left": 6,
    "Upper_Body_Right": 7,
    "Upper_Body_Left": 8
}

# %% 추론 데이터 로더 초기화 함수
def infer_dataloader(data_path, type = 'binary_classification', seq_len = 600, infer_bs = 4, infer_worker = 4, random_seed = 42):
    if type == 'binary_classification':
        infer_dataset = DK_Binary_Infer_Dataset(
            data_path = data_path,
            seq_length = seq_len,
            standardize = True
        )
    elif type == 'multi_classification':
        infer_dataset = DK_Multi_Infer_Dataset(
            data_path = data_path,
            seq_length = seq_len,
            standardize = True
        )

    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size = infer_bs,
        shuffle = False,
        drop_last = True,
        num_workers= infer_worker,
        pin_memory= True,
        worker_init_fn = lambda _: np.random.seed(random_seed)
    )
    
    return infer_dataloader

class DK_Binary_Infer_Dataset(Dataset):
    def __init__(self, data_path, seq_length, transform=None, standardize=False):
        self.data_path = data_path
        self.seq_length = seq_length
        self.transform = transform
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
        df= pd.read_csv(data_path)
        
        data_array = df.iloc[:, 0:].values
        num_frames = len(data_array)
        self.data = []
        if num_frames >= seq_length:
            for i in range(num_frames - seq_length + 1):
                self.data.append(data_array[i:i+seq_length])
        else:
            padding_needed = seq_length - num_frames
            padded_sequence = np.pad(data_array, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
            self.data.append(padded_sequence)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # PyTorch 텐서로 변환
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)
        return sample, label

class DK_Multi_Infer_Dataset(Dataset):
    def __init__(self, data_path, seq_length, transform=None, standardize=False):
        self.data_path = data_path
        self.seq_length = seq_length
        self.transform = transform
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        
        df= pd.read_csv(data_path)
        
        data_array = df.iloc[:, 0:].values
        num_frames = len(data_array)
        self.data = []
        if num_frames >= seq_length:
            for i in range(num_frames - seq_length + 1):
                self.data.append(data_array[i:i+seq_length])
        else:
            padding_needed = seq_length - num_frames
            padded_sequence = np.pad(data_array, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
            self.data.append(padded_sequence)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # PyTorch 텐서로 변환
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)
        return sample, label
        



#%%  훈련/검증 데이터 로더 초기화 함수
def init_dataloader(data_dir, fold_dir, fold_num, train_bs=32, test_bs=16, train_workers=8, test_workers=4, random_seed=42, seq_len=600, type='binary_classification'):
    if type == 'binary_classification':
        train_dataset = DK_Binary_Dataset(
            data_dir=data_dir,
            fold_csv=os.path.join(fold_dir, f'train_fold_{fold_num}.csv'),
            seq_length=seq_len,
            standardize=True  # 표준화 활성화
        )
        test_dataset = DK_Binary_Dataset(
            data_dir=data_dir,
            fold_csv=os.path.join(fold_dir, f'test_fold_{fold_num}.csv'),
            seq_length=seq_len,
            standardize=True  # 표준화 활성화
        )

    elif type == 'multi_classification':
        train_dataset = DK_Multi_Dataset(
            data_dir=data_dir,
            fold_csv=os.path.join(fold_dir, f'train_fold_{fold_num}.csv'),
            seq_length=seq_len,
            standardize=True  # 표준화 활성화
        )
        test_dataset = DK_Multi_Dataset(
            data_dir=data_dir,
            fold_csv=os.path.join(fold_dir, f'test_fold_{fold_num}.csv'),
            seq_length=seq_len,
            standardize=True  # 표준화 활성화
        )
        
        
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_bs, 
        shuffle=True, 
        drop_last=True, 
        num_workers=train_workers, 
        pin_memory=True, 
        worker_init_fn=lambda _: np.random.seed(random_seed)
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=test_bs, 
        shuffle=False, 
        drop_last=False, 
        num_workers=test_workers, 
        pin_memory=True, 
        worker_init_fn=lambda _: np.random.seed(random_seed)
    )
    
    return train_dataloader, test_dataloader

class DK_Binary_Dataset(Dataset):
    def __init__(self, data_dir, fold_csv, seq_length, transform=None, standardize=False):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.transform = transform
        self.standardize = standardize
        
        # 폴드 CSV 파일에서 파일명과 라벨을 읽어옵니다.
        self.file_labels = pd.read_csv(fold_csv)
        self.file_labels.reset_index(drop=True, inplace=True)
        
        # 표준화 준비
        self.scaler = StandardScaler() if standardize else None
        
        # 데이터 로드 및 전처리
        self.data = []
        self.labels = []
        for idx, row in self.file_labels.iterrows():
            filename = row['filename']
            label = row['label']
            
            # 데이터 파일 경로 설정
            feature_path = os.path.join(self.data_dir, filename + '_vel_acc.csv')
            
            # 데이터 로드
            try:
                combined_data = pd.read_csv(feature_path)
            except Exception as e:
                print(f"Error reading {feature_path}: {e}")
                continue
            
            # numpy 배열로 변환
            data_array = combined_data.values  # shape: (num_frames, num_features)
            
            # 표준화 적용
            if self.scaler:
                data_array = self.scaler.fit_transform(data_array)
            
            # 시퀀스 생성
            num_frames = len(data_array)
            if num_frames >= seq_length:
                # 충분한 길이의 시퀀스를 생성
                num_sequences = num_frames - seq_length + 1
                for i in range(num_sequences):
                    sequence = data_array[i:i + seq_length]
                    self.data.append(sequence)
                    
            else:
                # 데이터가 부족할 경우 패딩을 적용
                padding_needed = seq_length - num_frames
                padded_sequence = np.pad(data_array, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
                self.data.append(padded_sequence)
            
            # label을 시퀀스 길이만큼 반복하여 확장
            expanded_label = [label] * seq_length
            self.labels.append(expanded_label)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # PyTorch 텐서로 변환
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label
    


# %% [Classificaiton] Dataset
def multi_label_settings(POSE_GUIDE, select_video_df):
    """
        Traing / Testing 에서 multi label로 설정하기 위한 함수
    """
    labels_dict = {}
    labels = []
    for row in select_video_df.iterrows():
        filename = row[1]['filename']
        ROI = row[1]['ROI']
        if pd.isna(ROI): # ROI 가 nan인 경우
            labels_dict[str(filename)] = [0] * len(POSE_GUIDE)
        else:
            # 소문자로 만들기
            ROI = ROI.lower()
            ROI = ROI.replace('[', '').replace(']', '').split(',')
            ROI = [r.strip() for r in ROI]  # 공백 제거
            
            # 8개의 부위에 대해 0으로 초기화
            label = [0] * len(POSE_GUIDE)
            
            # ROI에 있는 각 부위에 대해 # One-Hot Encoding
            for roi in ROI:
                # POSE_GUIDE에서 해당 부위의 인덱스를 가져와서
                # 해당 인덱스의 label을 1로 설정
                if roi in POSE_GUIDE:
                    idx = list(POSE_GUIDE.keys()).index(roi)
                    label[idx] = 1
                    
            labels_dict[str(filename)] = label

    return labels_dict


class DK_Multi_Dataset(Dataset):
    def __init__(self, data_dir, fold_csv, seq_length=30, transform=None, standardize=False):
        self.pose_guide_dict = POSE_GUIDE
        # select_video_df = pd.read_csv('config/select_video_label.csv')
        # self.video_label_df = select_video_df
        self.labels_dict = multi_label_settings(self.pose_guide_dict, self.video_label_df) # 파일마다 라벨 딕셔너리 생성
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.transform = transform
        self.standardize = standardize
        # 폴드 CSV 파일에서 파일명과 라벨을 읽어옵니다.
        self.file_labels = pd.read_csv(fold_csv)
        self.file_labels.reset_index(drop=True, inplace=True)
        
        # 표준화 준비
        self.scaler = StandardScaler() if standardize else None
        
        # 데이터 로드 및 전처리
        self.data = []
        self.labels = []
        for idx, row in self.file_labels.iterrows():
            filename = row['filename']
            label = self.labels_dict[filename+'.mp4']
            
            # 데이터 파일 경로 설정
            feature_path = os.path.join(self.data_dir, filename + '_vel_acc.csv')
            
            # 데이터 로드
            try:
                combined_data = pd.read_csv(feature_path)
            except Exception as e:
                print(f"Error reading {feature_path}: {e}")
                continue
            
            # numpy 배열로 변환
            data_array = combined_data.values  # shape: (num_frames, num_features)
            
            # 표준화 적용
            if self.scaler:
                data_array = self.scaler.fit_transform(data_array)
            
            # 시퀀스 생성
            num_frames = len(data_array)
            if num_frames >= seq_length:
                # 충분한 길이의 시퀀스를 생성
                num_sequences = num_frames - seq_length + 1
                for i in range(num_sequences):
                    sequence = data_array[i:i + seq_length]
                    self.data.append(sequence)
            else:
                # 데이터가 부족할 경우 패딩을 적용
                padding_needed = seq_length - num_frames
                padded_sequence = np.pad(data_array, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
                self.data.append(padded_sequence)
                
            
            # label을 시퀀스 길이만큼 반복하여 확장
            expanded_label = [label] * seq_length
            self.labels.append(expanded_label)
            # self.labels.append(np.array(label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        # PyTorch 텐서로 변환
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label



class timesnet_freq_Dataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

    def __read_data__(self):
        self.scaler = StandardScaler()

        #get raw data from path
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))

        # split data set into train, vali, test. border1 is the left border and border2 is the right.
        # Once flag(train, vali, test) is determined, __read_data__ will return certain part of the dataset.
        border1s = [0, 
                    12 * 30 * 24 - self.seq_len, 
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len
                    ]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #decide which columns to select
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # column name list (remove 'date')
            df_data = df_raw[cols_data]  #remove the first column, which is time stamp info
        elif self.features == 'S':
            df_data = df_raw[[self.target]] # target column

        #scale data by the scaler that fits training data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            #train_data.values: turn pandas DataFrame into 2D numpy
            self.scaler.fit(train_data.values)  
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values 
        
        #time stamp:df_stamp is a object of <class 'pandas.core.frame.DataFrame'> and
        # has one column called 'date' like 2016-07-01 00:00:00
        df_stamp = df_raw[['date']][border1:border2]
        
        # Since the date format is uncertain across different data file, we need to 
        # standardize it so we call func 'pd.to_datetime'
        df_stamp['date'] = pd.to_datetime(df_stamp.date) 

        if self.timeenc == 0:  #time feature encoding is fixed or learned
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            #now df_frame has multiple columns recording the month, day etc. time stamp
            # next we delete the 'date' column and turn 'DataFrame' to a list
            data_stamp = df_stamp.drop(['date'], 1).values

        elif self.timeenc == 1: #time feature encoding is timeF
            '''
            when entering this branch, we choose arg.embed as timeF meaning we want to 
            encode the temporal info. 'freq' should be the smallest time step, and has 
            options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
            So you should check the timestep of your data and set 'freq' arg. 
            After the time_features encoding, each date info format will be encoded into 
            a list, with each element denoting  the relative position of this time point
            (e.g. Day of Week, Day of Month, Hour of Day) and each normalized within scope[-0.5, 0.5]
            '''
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        
        # data_x and data_y are same copy of a certain part of data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
