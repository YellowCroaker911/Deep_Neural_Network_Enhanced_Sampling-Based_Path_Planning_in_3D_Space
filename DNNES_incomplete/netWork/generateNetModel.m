function generateNetModel()
% 生成网络模型
    dataStore = loadData("trainData.mat");
    net = modelLayers();
    net = trainNet(net,dataStore);
    save("net.mat","net");
end

