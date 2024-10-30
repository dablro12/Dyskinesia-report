import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 


def visualize2img(data1, data2):
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(data1)
    plt.title('Input Image')
    plt.subplot(122)
    plt.imshow(data2)
    plt.title('Output Image')
    plt.tight_layout()
    plt.show()
    plt.close()
    
def loss_plotter(losses: list, save_path: str = None):
    # Seaborn 스타일 적용
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))

    # 일반 Loss Plot
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(len(losses)), y=losses, color='b', linewidth=1.5)
    plt.title('Loss Plot', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Log-scaled Loss Plot
    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(len(losses)), y=losses, color='orange', linewidth=1.5)
    plt.yscale('log')
    plt.title('Log Loss Plot', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (Log Scale)', fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 고해상도로 저장
    else:
        plt.show()
    plt.close()
    
def metric_plotter(best_metric_dict, save_path=None):
    fpr, tpr = best_metric_dict["fpr"], best_metric_dict["tpr"]
    conf_matrix = best_metric_dict["conf_matrix"]
    best_threshold = best_metric_dict["best_thr"]
    auc_score = best_metric_dict["best_auc"]
    
    plt.figure(figsize=(12, 6))
    
    # Confusion Matrix
    plt.subplot(121)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Dyskinesia Test Confusion Matrix")
    
    # 레이블 추가
    plt.xticks([0.5, 1.5], ['Off State', 'On State'])
    plt.yticks([0.5, 1.5], ['Off State', 'On State'])
    
    # ROC Curve
    plt.subplot(122)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    
    # Best Threshold 표시
    best_index = np.argmin(np.abs(fpr - best_threshold))
    plt.plot(fpr[best_index], tpr[best_index], 'ro', markersize=10, 
             label=f'Best Threshold ({best_threshold:.2f})')
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Dyskinesia Test ROC Curve")
    plt.legend()
    
    # AUC 점수 표시
    plt.text(0.05, 0.95, f'AUC Score: {auc_score:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()