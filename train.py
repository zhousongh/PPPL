import torch
from torch import nn
import numpy as np
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm
from model import Framework, HGNN
from dgllife.utils import RandomSplitter, ScaffoldSplitter
from dataset.dataset import FlexDataset, FlexDataLoader
from utils import Accumulator, get_loss_curve, get_roc_curve, load_model, getRightNum, getRMSE, get_scatter, print_item

# 参数
hidden_dim = 64
in_edge_width=7
in_node_width=100
num_epochs = 60
num_folds = 5
batch_size = 256
lr = 0.002
num_heads = 4
dataset_name = 'toxcast'
test_everyn = 1
task_num=12


device = torch.device("cuda:1")


def evaluate(net, test_set, task):
    # 测试
    net.eval()
    y_pred, y_true = [], []
    metric = Accumulator(2)
    test_loader = FlexDataLoader(
        device=device, dataset=test_set, batch_size=len(test_set), shuffle=False, drop_last=False)
    for G_batch, labels in test_loader:
        y_hat = None
        # 若数据有问题直接跳过
        try:
            y_hat = net(G_batch=G_batch)
        except Exception as e:
            print(e)
            continue

        y_hat = y_hat.view(labels.shape)

        if task == "classification":
            right_num = getRightNum(y_hat, labels)
            metric.add(right_num, float(labels.numel()))
        else:
            rmse = getRMSE(y_hat, labels)
            metric.add(rmse, 1)

        y_pred.extend(y_hat.tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred, metric[0]/metric[1]


def train_one_fold(train_set, val_set, net, loss, task, lr):
    '''训练一个fold的数据，返回更新后的模型和平均损失'''
    train_metric, test_metric, accu_metric = Accumulator(
        2), Accumulator(2), Accumulator(2)

    trainer = torch.optim.Adam(
        net.parameters(), lr=lr)
    # 训练
    net.train()
    train_loader = FlexDataLoader(
        device=device, dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = FlexDataLoader(
        device=device, dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=False)
    for G_batch, labels in train_loader:
        y_hat = None
        # 若数据有问题直接跳过
        try:
            y_hat = net(G_batch=G_batch)
        except Exception as e:
            print(e)
            continue
        y_hat = y_hat.view(labels.shape)
        l = loss(y_hat, labels)
        # 梯度下降
        trainer.zero_grad()
        l.backward()
        trainer.step()

        train_metric.add(float(l), float(labels.numel()))
    # 训练平均损失
    train_l = train_metric[0] / train_metric[1]

    # 测试
    net.eval()
    for G_batch, labels in val_loader:
        y_hat = None
        # 若数据有问题直接跳过
        try:
            # y_hat = net((G_batch, subG_batch))
            y_hat = net(G_batch=G_batch)
        except Exception as e:
            print(e)
            continue

        y_hat = y_hat.view(labels.shape)
        l = loss(y_hat, labels)

        if task == "classification":
            # 计算分类正确的样本数
            right_num = getRightNum(y_hat, labels)
            # 更新累加器
            accu_metric.add(float(right_num), float(labels.numel()))
        else:
            rmse = getRMSE(y_hat, labels)
            accu_metric.add(rmse, 1)

        test_metric.add(float(l), float(labels.numel()))
    # 测试的平均损失
    test_l = test_metric[0] / test_metric[1]
    # 分类的精度或平均RMSE
    accuracy = accu_metric[0] / accu_metric[1]
    return net, train_l, test_l, accuracy


def train(net, loss, train_folds, test_set, lr, task, save_path=r'/mnt/klj/PPPL/model/'):

    train_losses = []  # 存储训练损失值
    test_losses = []  # 存储测试损失值

    for epoch in tqdm(range(num_epochs)):
        train_metric, test_metric, accu_metric = Accumulator(
            2), Accumulator(2), Accumulator(2)  # 累加器

        for fold in train_folds:
            train_set, test_set = fold[0], fold[1]
            net, train_l, test_l, accuracy = train_one_fold(
                train_set=train_set, val_set=test_set, net=net, loss=loss, lr=lr, task=task)
            train_metric.add(float(train_l), 1)
            test_metric.add(float(test_l), 1)
            accu_metric.add(accuracy, 1)
        train_losses.append(train_metric[0]/train_metric[1])
        test_losses.append(test_metric[0]/test_metric[1])
        if (epoch+1) % test_everyn == 0:
            if task == "classification":
                _, _, test_accu = evaluate(net, test_set, task)
                print(
                    f'epoch{epoch + 1}:训练损失为 {print_item(train_losses[-1])}，验证损失为 {print_item(test_losses[-1])},验证集上的平均分类精度为{print_item(accu_metric[0]/accu_metric[1])},测试集上的平均分类精度为{print_item(test_accu)}')
            else:
                _, _, test_rmse = evaluate(net, test_set, task)
                print(
                    f'epoch{epoch + 1}:训练损失为 {print_item(train_losses[-1])}，验证损失为 {print_item(test_losses[-1])},验证集上的平均RMSE为{print_item(accu_metric[0]/accu_metric[1])},测试集上的平均RMSE为{print_item(test_rmse)}')
    get_loss_curve(train_losses, test_losses, dataset=dataset_name)
    y_true, y_pred, _ = evaluate(net, test_set, task=task)
    if task == "classification":
        get_roc_curve(y_true=y_true, y_pred=y_pred, dataset=dataset_name)
    else:
        get_scatter(y_true=y_true, y_pred=y_pred, dataset=dataset_name)
    torch.save(net.state_dict(), f'{save_path}{dataset_name}_model_params.pth')
    return net


if __name__ == "__main__":

    # 加载数据集
    dataset = FlexDataset(dataset_name=dataset_name,
                          root=r'/mnt/klj/PPPL/dataset/data', device=device)
    # 划分数据集：随机划分或骨架划分
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(dataset)
    # train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(dataset)
    train_folds = RandomSplitter.k_fold_split(
        ConcatDataset([train_set, val_set]))

    # 定义模型和损失函数
    classifier = nn.Sequential(
    nn.Linear(hidden_dim, dataset[0]["label"].numel()), nn.Sigmoid())
    model = Framework(in_node_features=in_node_width, in_edge_features=in_edge_width,
                          hidden_dim=hidden_dim, aggr='sum', residual=False, num_heads=num_heads, predictor=classifier,task_num=task_num,device=device).to(device)
    loss = nn.BCELoss(reduction='sum', weight=dataset.task_pos_weights())
    task = 'classification'

    # 是否加载预训练模型
    # model = load_model(model)

    model = train(net=model, loss=loss, train_folds=train_folds,
                  test_set=test_set, lr=lr, task=task)
