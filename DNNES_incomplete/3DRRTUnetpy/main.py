from torchsummary import summary

from my_dataset import *
from my_u_net import *
from my_training_loop import *

# 查看网络架构
net = MyUnet()
net = net.to("cuda")
summary(net,[(1,32,32,32),(1,32,32,32)])

# lr, num_epochs, batch_size = 0.001, 100, 1      # 学习率（固定），迭代周期，批大小
# train_iter, test_iter = load_data(batch_size)   # 训练数据迭代器，测试数据迭代器
# train(net,train_iter,test_iter,num_epochs,lr,device=d2l.try_gpu())  # 训练网络
# d2l.plt.show()                                                      # 查看迭代结果
#
# # 保存网络模型
# torch.save(net.state_dict(), 'net.pth')