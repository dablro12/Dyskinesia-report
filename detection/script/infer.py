# Description: This script is used to load the checkpoint and return the model weight and config.
import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# %% [Load Libraries]
import torch
from dataset.custom_dataset import infer_dataloader
from utils.transform import reshape_tensor
from model import init_model

def load_ckpt(ckpt):
    ckpt = torch.load(ckpt)
    weight = ckpt['model_state_dict']
    _ = ckpt['model_size']
    config = ckpt['config']
    __ = ckpt['metric_dict']
    best_thr = ckpt['best_threshold']
    return weight, config, best_thr


# %% [ML Classifier] Module
def ml_classifier(ckpt_path, data_path):
    weight, config, best_thr = load_ckpt(ckpt_path)
    
    infer_loader = infer_dataloader(
        data_path = data_path,
        type = config["task_type"],
        seq_len = config["seq_len"],
        infer_bs = 1,
        infer_worker= 4,
        random_seed= 42
    )
    X_infer, _ = reshape_tensor(infer_loader)
    model, config = init_model(X_infer, _, config, bayes_opt= False)
    pred = model.inference_forward(reshape_X = X_infer, weight = weight, thr = best_thr)
    # Pred 를 print 문으로 잘보이게 배경색과 글자색을 변경
    print("\033[48;5;230m" + f"\033[38;5;214m[Result] : {pred[0]}\033[0m")
    

    return pred[0] # Prediction
# %% [Usage] Prediction
# python script/infer.py --ckpt data/result/classifier/timesnet/vel_acc_kalman_symptom_1.pth --data_path /home/eiden/eiden/pd-ai/data/detection_data/vel_acc_kalman_symptom/1-1_9_vel_acc.csv
if __name__ == "__main__":
    import argparse
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--ckpt', type=str, help='checkpoint path')
    argparse.add_argument('--data_path', type=str, help='data_csv path')
    
    args = argparse.parse_args()
    weight, config, best_thr = load_ckpt(args.ckpt)

    infer_loader = infer_dataloader(
        data_path = args.data_path,
        type = config["task_type"],
        seq_len = config["seq_len"],
        infer_bs = 1,
        infer_worker= 4,
        random_seed= 42
    )
    X_infer, _ = reshape_tensor(infer_loader)
    model, config = init_model(X_infer, _, config, bayes_opt= False)
    pred = model.inference_forward(reshape_X = X_infer, weight = weight, thr = best_thr)
    # Pred 를 print 문으로 잘보이게 배경색과 글자색을 변경
    print("\033[48;5;230m" + f"\033[38;5;214m[Result] : {pred[0]}\033[0m")
    
# %% [Example Usage]
# EXAMPLE_DATA_PATH="/home/eiden/eiden/pd-ai/data/detection_data/vel_acc_kalman_symptom/1-1_9_vel_acc.csv"
# EXAMPLE_CKPT_PATH="data/result/classifier/timesnet/vel_acc_kalman_symptom_1.pth"
# EXAMPLE_CKPT_PATH="/home/eiden/eiden/pd-ai/data/result/classifier/timesnet/vel_acc_kalman_symptom_multi_classification_4.pth"
