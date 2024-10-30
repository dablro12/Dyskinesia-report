from sklearn.model_selection import StratifiedKFold, KFold
def k_fold_split_by_patient(df, n_splits=4, random_state=42):
    """
    환자 데이터를 첫 번째 숫자 기준으로 그룹핑하고, label 비율을 유지하지 않으면서 k-fold로 분리하는 함수.
    
    Parameters:
    - df (pd.DataFrame): 데이터프레임, 'filename'과 'label' 컬럼이 있어야 함.
    - n_splits (int): k-fold의 k 값 (기본값: 4)
    - random_state (int): 랜덤 시드 (기본값: 42)
    
    Yields:
    - train_df (pd.DataFrame): train 데이터프레임
    - test_df (pd.DataFrame): test 데이터프레임
    """
    
    # filename의 첫 번째 숫자 추출하여 새로운 컬럼 추가
    df['patient_id'] = df['filename'].str.split('-').str[0]

    # 환자별 그룹핑
    grouped = df.groupby('patient_id').agg({'label': 'first'}).reset_index()
    
    # KFold을 사용하여 비율 유지 없이 k-fold split
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold = 1
    for train_index, test_index in kf.split(grouped['patient_id']):
        train_patients = grouped.loc[train_index, 'patient_id'].tolist()
        test_patients = grouped.loc[test_index, 'patient_id'].tolist()
        
        # train/test 데이터프레임으로 분리
        train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
        test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

        # filename과 label만 가지고 옴
        train_df = train_df[['filename', 'label']]
        test_df = test_df[['filename', 'label']]
        
        # filename에서 .mp4 제거
        train_df['filename'] = train_df['filename'].str.replace('.mp4', '')
        test_df['filename'] = test_df['filename'].str.replace('.mp4', '')
        # Test 환자 ID 출력
        print(f"Fold {fold}: Train patients: {train_patients}, Test patients: {test_patients}")
        yield fold, train_df, test_df
        fold += 1


def k_fold_stratified_split_by_patient(df, n_splits=4, random_state=42):
    """
    환자 데이터를 첫 번째 숫자 기준으로 그룹핑하고, label 비율을 유지하면서 k-fold로 분리하는 함수.
    
    Parameters:
    - df (pd.DataFrame): 데이터프레임, 'filename'과 'label' 컬럼이 있어야 함.
    - n_splits (int): k-fold의 k 값 (기본값: 4)
    - random_state (int): 랜덤 시드 (기본값: 42)
    
    Yields:
    - train_df (pd.DataFrame): train 데이터프레임
    - test_df (pd.DataFrame): test 데이터프레임
    """
    
    # filename의 첫 번째 숫자 추출하여 새로운 컬럼 추가
    df['patient_id'] = df['filename'].str.split('-').str[0]

    # 환자별 그룹핑
    grouped = df.groupby('patient_id').agg({'label': 'first'}).reset_index()
    
    # StratifiedKFold을 사용하여 label 비율을 유지하면서 k-fold split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold = 1
    for train_index, test_index in skf.split(grouped['patient_id'], grouped['label']):
        train_patients = grouped.loc[train_index, 'patient_id'].tolist()
        test_patients = grouped.loc[test_index, 'patient_id'].tolist()
        
        # train/test 데이터프레임으로 분리
        train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
        test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

        # filename과 label만 가지고 옴
        train_df = train_df[['filename', 'label']]
        test_df = test_df[['filename', 'label']]
        # filename에서 .mp4 제거
        train_df['filename'] = train_df['filename'].str.replace('.mp4', '')
        test_df['filename'] = test_df['filename'].str.replace('.mp4', '')
        # Test 환자 ID 출력
        print(f"Fold {fold}: Train patients: {train_patients}, Test patients: {test_patients}")
        yield fold, train_df, test_df
        fold += 1