import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
# %% [TimesNet] Train
def trainer(model , X_train, y_train):
    losses = model.fit(X_train, y_train)
    return model, losses
# %% [LSTM, Transformer] Trainer Class
class ParkinsonsTrainer:
    def __init__(self, model, fold, train_loader, test_loader, criterion, optimizer, save_dir, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.fold = fold
        self.optimizer = optimizer
        self.device = device
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader

        self.save_dir = os.path.join(save_dir, f"fold_{fold}")
        os.makedirs(self.save_dir, exist_ok=True)

        # 학습 기록 저장을 위한 초기화
        self.training_history = {
            'epoch_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc': []
        }

    # 모델 학습 메서드
    def train(self, num_epochs=100, patience=50):
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, metrics = self.evaluate()

            # 학습 기록 저장
            self.training_history['epoch_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['accuracy'].append(metrics['accuracy'])
            self.training_history['precision'].append(metrics['precision'])
            self.training_history['recall'].append(metrics['recall'])
            self.training_history['f1_score'].append(metrics['f1'])
            self.training_history['auc'].append(metrics['auc'])

            # 에포크별 결과 출력
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # 조기 중단 로직
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_model(f'epoch_{epoch}_best.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

        # 마지막 모델 및 평가 지표 저장
        self.save_model(f'epoch_{epoch}.pth')
        self.save_metrics(metrics, epoch)
        self.plot_roc_curve(metrics['labels'], metrics['probs'], epoch)
        self.plot_confusion_matrix(metrics['labels'], metrics['predictions'], epoch)

    # 에포크별 학습 메서드
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for samples, labels in self.train_data_loader:
            samples, labels = samples.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # 모델 예측 및 손실 계산
            outputs = self.model(samples)
            loss = self.criterion(outputs, labels)

            # 역전파 및 최적화
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        # 에포크의 평균 손실 반환
        return epoch_loss / len(self.train_data_loader)

    # 모델 평가 메서드
    def evaluate(self):
        self.model.eval()
        eval_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for samples, labels in self.test_data_loader:
                samples, labels = samples.to(self.device), labels.to(self.device)

                # 모델 예측
                outputs = self.model(samples)
                loss = self.criterion(outputs, labels)
                eval_loss += loss.item()

                # 예측 라벨 및 확률 계산
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        # 평가 지표 계산
        metrics = self.calculate_metrics(all_labels, all_predictions, all_probs)
        return eval_loss / len(self.test_data_loader), metrics

    # 평가 지표 계산 메서드
    def calculate_metrics(self, labels, predictions, probs):
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'auc': roc_auc_score(labels, probs),
            'labels': labels,
            'predictions': predictions,
            'probs': probs
        }

    # 모델 저장 메서드
    def save_model(self, file_name):
        save_path = os.path.join(self.save_dir, file_name)
        torch.save(self.model.state_dict(), save_path)
        print("\033[48;5;230m" + f"\033[38;5;214m[Model Save] : {save_path}\033[0m")

    # ROC Curve 그리기 메서드
    def plot_roc_curve(self, labels, probs, epoch):
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Epoch {epoch}')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch}_roc_curve.png'))
        plt.close()
        print("\033[48;5;230m" + f"\033[38;5;214m[ROC Curve Save] : epoch_{epoch}_roc_curve.png\033[0m")

    # 평가 메트릭을 텍스트 파일로 저장
    def save_metrics(self, metrics, epoch):
        metrics_path = os.path.join(self.save_dir, f'epoch_{epoch}_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f'Epoch: {epoch}\n')
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"AUC: {metrics['auc']:.4f}\n")
        print("\033[48;5;230m" + f"\033[38;5;214m[Metrics Save] : {metrics_path}\033[0m")

    # 혼동 행렬 그리기 메서드
    def plot_confusion_matrix(self, labels, predictions, epoch):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch}_confusion_matrix.png'))
        plt.close()
        print("\033[48;5;230m" + f"\033[38;5;214m[Confusion Matrix Save] : epoch_{epoch}_confusion_matrix.png\033[0m")
