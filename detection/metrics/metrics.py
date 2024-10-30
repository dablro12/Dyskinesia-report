import numpy as np
from sklearn import metrics

def accuracy_score(y_true, y_pred):
    """
    정확도 계산
    """
    return metrics.accuracy_score(y_true, y_pred)
def auc_score(y_true, y_score):
    """
    ROC curve 아래 면적으로 AUC score 계산
    """
    return metrics.roc_auc_score(y_true, y_score)

# Recall Score 계산 함수 정의
def recall_score(y_true, y_pred):
    """
    Recall Score 계산
    """
    return metrics.recall_score(y_true, y_pred)

def pr_score(y_true, y_score): # 함수 이름 바꾸기 auc 아님!!
    """
    Precision-Recall curve 아래 면적 계산
    """
    return metrics.average_precision_score(y_true, y_score)


def point_adjustment(y_true, y_score):
    """
    * [Zhihan Li et al. KDD21]의 코드를 참고, 수정함 *
    이상 데이터가 발생하면 해당 시점으로부터 특정 기간동안 지속되는 특징에 따라
    해당 타임 포인트의 최댓값으로 내부 값들을 재정의
    """
    score = y_score.copy()
    assert len(score) == len(y_true)
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = y_true[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


def get_best_f1(label, score):
    """
    가장 높은 f1 점수 기록
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    return best_f1, best_p, best_r


def ts_metrics(y_true, y_score):
    """
    AUC score, Precision-Recall score, f1, precision, recall 계산
    """
    best_f1, best_p, best_r = get_best_f1(y_true, y_score)
    best_acc = accuracy_score(y_true, y_score > 0.5)
    return auc_score(y_true, y_score), best_acc, pr_score(y_true, y_score), best_f1, best_p, best_r



def test_metric(label_arr, prob_arr,  thr_li = [0.1, 0.25, 0.5, 0.75, 0.9], n_features = 1):
    """test_metric

    Args:
        label_arr (_array_): array of label
        prob_arr (_array_): array of probability
        thr_li (_list_): threshold example [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    print(f"[test metric func] label_arr shpae : {label_arr.shape}, prob_arr shape : {prob_arr.shape}")
    
    if n_features == 1: #Binary Classification
        # 메트릭 뽑기 
        f1 = []
        precision = []
        recall = []
        accuracy = []
        auc = []
        
        for th in thr_li:
            pred_arr = prob_arr > th
            f1.append(metrics.f1_score(label_arr, pred_arr))
            precision.append(metrics.precision_score(label_arr, pred_arr))
            recall.append(metrics.recall_score(label_arr, pred_arr))
            accuracy.append(metrics.accuracy_score(label_arr, pred_arr))
            auc.append(metrics.roc_auc_score(label_arr, prob_arr))
        # 가장 좋은 메트릭의 threshold 를 정하고 뽑기
        best_f1 = max(f1)
        best_f1_idx = f1.index(best_f1)
        best_precision = precision[best_f1_idx]
        best_recall = recall[best_f1_idx]
        best_accuracy = accuracy[best_f1_idx]
        best_auc = auc[best_f1_idx]
        best_thr = thr_li[best_f1_idx]
        
        # Confusion Matrix 생성
        pred_arr = prob_arr > best_thr
        conf_matrix = metrics.confusion_matrix(label_arr, pred_arr)
        # ROC Curve 생성
        fpr, tpr, _ = metrics.roc_curve(label_arr, prob_arr)
        
        
        best_metric_dict = {
            "best_f1" : best_f1,
            "best_precision" : best_precision,
            "best_recall" : best_recall,
            "best_accuracy" : best_accuracy,
            "best_auc" : best_auc,
            "best_thr" : best_thr,
            "conf_matrix" : conf_matrix,
            "fpr" : fpr,
            "tpr" : tpr
        }
        return best_metric_dict
    elif n_features > 1:  # Multi Classification
        # 각 threshold별로 메트릭 계산
        precision_list = []
        recall_list = []
        accuracy_list = []
        f1_list = []
        auc_list = []
        for th in thr_li:
            # threshold 적용하여 예측값 생성
            pred_arr = (prob_arr > th).astype(int)
            
            # 각 메트릭 계산 pred_arr, label_arr 는 [n_samples, n_features] 형태
            # 각 메트릭 계산
            precision = metrics.precision_score(label_arr, pred_arr, average='macro')
            recall = metrics.recall_score(label_arr, pred_arr, average='macro')
            accuracy = metrics.accuracy_score(label_arr, pred_arr)
            f1 = metrics.f1_score(label_arr, pred_arr, average='macro')
            auc = metrics.roc_auc_score(label_arr, prob_arr, average='macro', multi_class='ovo')
            
            precision_list.append(precision)
            recall_list.append(recall)
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            auc_list.append(auc)

        # 가장 좋은 F1 score를 기준으로 best threshold 선정
        best_idx = np.argmax(f1_list)  # 여기서 precision을 기준으로 최적의 인덱스를 찾음
        best_thr = thr_li[best_idx]

        # 최종 예측값 생성
        pred_arr = prob_arr > best_thr
        conf_matrix = metrics.confusion_matrix(label_arr, pred_arr)
        fpr, tpr, _ = metrics.roc_curve(label_arr, prob_arr)

        # 최종 메트릭 딕셔너리 생성
        best_metric_dict = {
            "best_f1": f1_list[best_idx],
            "best_precision": precision_list[best_idx],
            "best_recall": recall_list[best_idx],
            "best_accuracy": accuracy_list[best_idx],
            "best_auc": auc_list[best_idx],
            "best_thr": best_thr,
            "conf_matrix" : conf_matrix,
            "fpr" : fpr,
            "tpr" : tpr
        }

        return best_metric_dict