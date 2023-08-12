import torch
import numpy as np
from interact.module import MyUnet


def prediction(params_path, X1,X2):
    """
    代码说明请参考https://www.bilibili.com/video/BV1WY4y17728，https://zhuanlan.zhihu.com/p/536858806
    """
    model = MyUnet()
    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    model.eval()
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    X1 = torch.from_numpy(X1)
    X2 = torch.from_numpy(X2)
    X1 = X1.type(torch.FloatTensor)
    X2 = X2.type(torch.FloatTensor)
    """
        由于matlab的数据格式为SSSCB（即数据维，数据维，数据维，通道维，批量维）
        而pytorch的数据格式为BCSSS，故需要通过permuteH函数转化
    """
    X1 = torch.Tensor(X1).permute(4, 3, 0, 1, 2)
    X2 = torch.Tensor(X2).permute(4, 3, 0, 1, 2)

    with torch.no_grad():
        pred = model(X1,X2)
    pred = torch.Tensor(pred).permute(2, 3, 4, 1, 0)
    pred = pred.detach().numpy()
    pred = np.ascontiguousarray(pred)

    return pred