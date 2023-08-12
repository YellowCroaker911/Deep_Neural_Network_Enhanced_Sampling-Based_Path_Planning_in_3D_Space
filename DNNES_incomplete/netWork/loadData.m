function ds = loadData(name)
% 将路径name中的.mat文件转为TransformDataStore类型（这样才能传入matlab的神经网络）
    fds = fileDatastore(name,ReadFcn=@load);
    ds = transform(fds,@transformFcn);
end

function dsNew = transformFcn(ds1)
    dsNew = ds1.dataCell;
end
