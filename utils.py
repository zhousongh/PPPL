import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from torch import nn
import numpy as np
import torch


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_roc_curve(y_true, y_pred, dataset, save_path=r'/mnt/klj/PPPL/pictures/'):
    '''绘制ROC曲线，并将图片保存至指定路径'''
    plt.figure()
    roc_auc=[]
    y_true, y_pred = torch.tensor(y_true).t(), torch.tensor(y_pred).t()
    color = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#ff8884', '#BB9727',
            '#54B345', '#32B897', '#05B9E2', '#8983BF', '#C76DA2', '#9E9E9E']
    for i in range(y_true.shape[0]):
        # 计算fpr和tpr
        fpr, tpr, _ = roc_curve(y_true[i].tolist(), y_pred[i].tolist())
        # 计算auc值
        roc_auc.append(auc(fpr, tpr))

        plt.plot(
                fpr, tpr, color=color[i], lw=1, label=f'ROC{i+1} curve')
    roc_auc = sum(roc_auc)/len(roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve (AUC = {roc_auc:.2f})')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_path}{dataset}_ROC.png')


def get_scatter(y_true, y_pred, dataset, save_path=r'/mnt/klj/PPPL/pictures/'):
    plt.figure()
    # 绘制预测值和真实值之间的散点图
    plt.scatter(x=list(range(1, len(y_true)+1)), y=y_true, label='True Values',
                color='b', marker='o', s=20)
    plt.scatter(x=list(range(1, len(y_pred)+1)), y=y_pred, label='Predictions',
                color='red', marker='o', s=20)
    plt.title('Regression Model Performance')
    plt.xlabel('Sameple No.')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}{dataset}_scatter.png')


def get_loss_curve(train_losses, test_losses, dataset, save_path=r'/mnt/klj/PPPL/pictures/'):
    '''绘制loss曲线,train_losses为一个列表，每一项为每个epoch的平均损失，test_losses同'''
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig(f'{save_path}{dataset}_loss.png')


def load_model(model, dataset, save_path='/mnt/klj/PPPL/model/'):
    model.load_state_dict(torch.load(f'{save_path}{dataset}_model_params.pth'))
    return model


def getRightNum(y_hat, labels):
    '''计算分类正确的样本数'''
    assert y_hat.shape == labels.shape
    y1, y2 = y_hat, labels
    y1[y1 >= 0.5] = 1
    y1[y1 < 0.5] = 0
    cmp = y1 == y2
    return cmp.type(y2.dtype).sum()


def getRMSE(y_hat, y_true):
    '''计算RMSE'''
    residuals = y_hat - y_true
    return torch.sqrt(torch.mean(residuals**2)).item()

def print_item(x):
    if torch.is_tensor(x):
        return x.item()
    else:
        return x