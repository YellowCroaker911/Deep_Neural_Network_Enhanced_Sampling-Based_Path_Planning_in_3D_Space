import torch
from d2l import torch as d2l
from torch import nn


def train(net, train_iter, test_iter, num_epochs, lr, device):
    """
    代码细节参考李沐《动手学深度学习》B站视频，此函数参照d2l.train_ch6修改（对应书上第6章的训练函数）
    :param net: 网络模型
    :param train_iter: 训练数据dataloader
    :param test_iter:  测试数据dataloader
    :param num_epochs: 迭代周期数
    :param lr: 学习率
    :param device: 设备（cpu或cuda）
    """

    # 权重初始化（卷积和转置卷积层使用HeKaiming初始化）
    def init_weights(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)                # 定义优化策略
    loss = nn.BCELoss()                                                 # 定义损失函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],       # 动画窗口初始化
                            legend=['train loss', 'test loss'])
    timer, num_batches = d2l.Timer(), len(train_iter)                   # 时间戳，批数量
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(1)                                     # 定义动画信息累加器
        net.train()                                                     # 网络模型设为训练模式
        for i, (X1, X2, T) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X1, X2, T = X1.unsqueeze(1), X2.unsqueeze(1), T.unsqueeze(1)    # 令X1,X2,T维度大小为1的维度不压缩，保证维度数量不变
            X1, X2, T = X1.to(device), X2.to(device), T.to(device)
            Y = net(X1, X2)
            l = loss(Y, T)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l)
            train_l = metric[0]                                             # 训练损失
            timer.stop()
            if (i + 1) % (num_batches // 1) == 0 or i == num_batches - 1:   # 每间隔1次进行一次网络测试
                animator.add(epoch + (i + 1) / num_batches,(train_l, None))
        test_l = evaluate(net, test_iter)                                   # 测试损失
        animator.add(epoch + 1, (None, test_l))
    print(f'train_loss {train_l:.3f} '
          f'test loss {test_l:.3f}')
    print(f'{metric[0] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def evaluate(net, data_iter, device=None):
    """
    返回测试（评价）的损失
    """
    if isinstance(net, nn.Module):
        net.eval()                                                      # 网络模型设为评价模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(1)

    with torch.no_grad():
        for X1,X2,T in data_iter:
            if isinstance(X1,list) and isinstance(X2,list):
                X1, X2 = [x.to(device) for x in X1], [x.to(device) for x in X2]
            else:
                X1, X2, = X1.unsqueeze(1), X2.unsqueeze(1)
                X1, X2 = X1.to(device), X2.to(device)
            T = T.unsqueeze(1)
            T = T.to(device)
            metric.add(getloss(net(X1,X2), T))
    return metric[0]


def getloss(Y,T):
    loss = nn.BCELoss()
    l = loss(Y,T)
    return l
