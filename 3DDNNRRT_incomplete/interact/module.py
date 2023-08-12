import torch
from torch import nn


class MyConv3dBlock(nn.Module):
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=input_channels,
                              out_channels=num_channels,
                              kernel_size=(3, 3, 3),
                              stride=(1, 1, 1),
                              padding =(1, 1, 1))
        self.mp = nn.MaxPool3d(kernel_size=(2, 2, 2),
                               stride=(2, 2, 2))
        self.bn = nn.BatchNorm3d(num_channels)
        self.rl = nn.ReLU()

    def forward(self, X):
        Y = self.rl(self.bn(self.mp(self.conv(X))))
        return Y


class MyConvTransPose3dBlock(nn.Module):
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.tpconv = nn.ConvTranspose3d(in_channels=input_channels,
                                         out_channels=num_channels,
                                         kernel_size=(2, 2, 2), # origin paper论文是3*3*3，但如此无法满足输出大小恰好翻倍
                                         stride=(2, 2, 2),
                                         padding=(0, 0, 0))
        self.bn = nn.BatchNorm3d(num_channels)
        self.rl = nn.ReLU()

    def forward(self, X):
        Y = self.rl(self.bn(self.tpconv(X)))
        return Y


class MyUnet(nn.Module):
    def __init__(self):

        # encoder
        super().__init__()
        self.my_conv3d1a = MyConv3dBlock(1,16)
        self.my_conv3d1b = MyConv3dBlock(1,16)
        self.my_conv3d2 = MyConv3dBlock(32,64)
        self.my_conv3d3 = MyConv3dBlock(64,128)
        self.my_conv3d4 = MyConv3dBlock(128,256)

        # decoder
        self.my_tpconv3d1 = MyConvTransPose3dBlock(256,128)
        self.my_tpconv3d2 = MyConvTransPose3dBlock(128,64)
        self.my_tpconv3d3 = MyConvTransPose3dBlock(64, 32)
        self.my_tpconv3d4 = MyConvTransPose3dBlock(32, 16)
        self.conv1x1 = nn.Conv3d(16, 1, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0))
        self.sig = nn.Sigmoid()


    def forward(self, x1, x2):
        # encoder
        out1a = self.my_conv3d1a(x1)
        out1b = self.my_conv3d1b(x2)
        out1ab = torch.cat([out1a, out1b], dim=1)
        out2 = self.my_conv3d2(out1ab)
        out3 = self.my_conv3d3(out2)
        out4 = self.my_conv3d4(out3)

        # decoder
        out5 = self.my_tpconv3d1(out4)
        out6 = self.my_tpconv3d2(out5 + out3)
        out7 = self.my_tpconv3d3(out6 + out2)
        out8 = self.my_tpconv3d4(out7 + out1ab)
        out9 = self.conv1x1(out8)
        out10 = self.sig(out9)

        return out10
