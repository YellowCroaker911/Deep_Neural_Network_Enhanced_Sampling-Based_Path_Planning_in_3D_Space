import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io

class MyDataset(Dataset):
    """
    自定义2输入1输出的dataloader类型
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input1, input2, output = self.data[index]
        return torch.Tensor(input1), torch.Tensor(input2), torch.Tensor(output)


def load_data(batch_size,dataloader_workers=0):
    """
    从.mat文件中加载数据，转为dataloader类型
    """
    mat_train = scipy.io.loadmat('../trainData.mat')
    mat_test = scipy.io.loadmat('../testData.mat')
    mat_train = mat_train['dataCell']
    mat_test = mat_test['dataCell']
    data_train = MyDataset(mat_train)
    data_test = MyDataset(mat_test)
    return (DataLoader(data_train,batch_size,shuffle=True,num_workers=dataloader_workers),
            DataLoader(data_test,batch_size,shuffle=True,num_workers=dataloader_workers))
