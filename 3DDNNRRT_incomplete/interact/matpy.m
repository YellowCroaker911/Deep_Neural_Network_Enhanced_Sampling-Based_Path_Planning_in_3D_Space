function pred = matpy(params_path, X1,X2)
% 输入：params_path为网络模型权重文件路径，X1,X2为网络模型的输入
    net = py.importlib.import_module('interact.module');
    py.importlib.reload(net);
    model = py.importlib.import_module('interact.pred');
    py.importlib.reload(model);
    pred = model.prediction(pyargs('params_path', params_path, 'X1', X1, 'X2', X2));
    pred = double(pred);
end