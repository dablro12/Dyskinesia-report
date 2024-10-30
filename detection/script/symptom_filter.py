import pandas as pd 
import os
import json 
class symtom_fitler:
    def __init__(self, vel_acc_dir, pose_guide_path, symptom_pos_guide_save_path, symptom_pose_df_save_path, save = False):
        self.vel_acc_dir = vel_acc_dir
        self.pose_guide_path = pose_guide_path
        self.symptom_pos_guide_save_path = symptom_pos_guide_save_path
        self.symptom_pose_df_save_path = symptom_pose_df_save_path
        self.symptom_position = self.get_symptom(save)
        
    def get_symptom(self, save = False):
        with open(self.pose_guide_path, 'r') as f:
            pose_guide = json.load(f)
        symptom_reference = ['Foot_Left', 'Foot_Right', 'Arm_Left', 'Arm_Right', 'Lower_Body_Right', 'Lower_Body_Left', 'Upper_Body_Right', 'Upper_Body_Left']
        # pose_guide에서 symptom_reference에 해당하는 key값을 찾아서 dict로 변환
        symptom_position = {key:pose_guide[key] for key in symptom_reference}
        # json 으로 저장
        if save:
            with open(self.symptom_pos_guide_save_path, 'w') as f:
                json.dump(symptom_position, f)
        return symptom_position

    def symptom_filter(self, vel_df, symptom_position, type):
        rows_list = []
        for row in vel_df.iterrows():
            row = row[1]
            
            row_dict = {}
            for key, value in symptom_position.items():
                for position in value:
                    x = f"{position}_x_{type}"
                    y = f"{position}_y_{type}"
                    z = f"{position}_z_{type}"
                    
                    row_dict[x] = row[x]
                    row_dict[y] = row[y]
                    row_dict[z] = row[z]
                    
            # Append the row dictionary to the list
            rows_list.append(row_dict)

        # Convert the list of rows into a DataFrame
        symptom_df = pd.DataFrame(rows_list)
        # 칼럼 오름차 순 
        symptom_df = symptom_df.reindex(sorted(symptom_df.columns), axis=1)

        # 칼럼에서 중복되는 값 제거
        symptom_df = symptom_df.loc[:,~symptom_df.columns.duplicated()]
        return symptom_df 

    def get_symptom_position(self, save = False):
        for file in os.listdir(self.vel_acc_dir):
            csv_path = os.path.join(self.vel_acc_dir, file)
            df = pd.read_csv(csv_path)
            vel_df = df[df.columns[:99]]
            acc_df = df[df.columns[99:]]
            
            vel_symptom_df = self.symptom_filter(vel_df, self.symptom_position, 'v')
            acc_symptom_df = self.symptom_filter(acc_df, self.symptom_position, 'a')

            # 두개 합치기
            symptom_df = pd.concat([vel_symptom_df, acc_symptom_df], axis=1)
            # 칼럼에서 중복되는 값 제거하고 칼럼 오름차순
            symptom_df = symptom_df.loc[:,~symptom_df.columns.duplicated()]
            symptom_df = symptom_df.reindex(sorted(symptom_df.columns), axis=1)
            
            if save:
                symptom_df.to_csv(self.symptom_pose_df_save_path + f'/{file}', index = False)
        
        return self.symptom_position
    
# %% Example Filter Symptom
if __name__ == "__main__":
    vel_acc_dir = '/home/eiden/eiden/pd-ai/data/detection_data/vel_acc_kalman'
    pose_guide_path = '/home/eiden/eiden/pd-ai/data/BACKUP/mediapipe_pose_guide.json'
    symptom_pos_guide_save_path = '/home/eiden/eiden/pd-ai/data/BACKUP/symptom_pos_guide.json'
    symptom_pose_df_save_path = "/home/eiden/eiden/pd-ai/data/detection_data/vel_acc_kalman_symptom"

    pose_filter = symtom_fitler(
        vel_acc_dir = vel_acc_dir,
        pose_guide_path = pose_guide_path, 
        symptom_pos_guide_save_path = symptom_pos_guide_save_path,
        symptom_pose_df_save_path = symptom_pose_df_save_path, 
        save = False
    )
    pose_filter.get_symptom_position(save = True) 