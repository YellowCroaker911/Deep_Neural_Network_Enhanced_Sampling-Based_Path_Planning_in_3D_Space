function net = modelLayers()
% 定义网络模型（可以用analyzeNetwork(net)查看结构）
lgraph = layerGraph();

tempLayers = [
    image3dInputLayer([32 32 32 1],"Name","Environment Map","Normalization","none")
    convolution3dLayer([3 3 3],16,"Name","conv3d_1","Padding",[1 1 1;1 1 1],"WeightsInitializer","he")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_1","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    image3dInputLayer([32 32 32 1],"Name","Start & Goal","Normalization","none")
    convolution3dLayer([3 3 3],16,"Name","conv3d_2","Padding",[1 1 1;1 1 1],"WeightsInitializer","he")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_2","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],64,"Name","conv3d_3","Padding",[1 1 1;1 1 1],"WeightsInitializer","he")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_3","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],128,"Name","conv3d_4","Padding",[1 1 1;1 1 1],"WeightsInitializer","he")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_4","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution3dLayer([3 3 3],256,"Name","conv3d_5","Padding",[1 1 1;1 1 1],"WeightsInitializer","he")
    maxPooling3dLayer([2 2 2],"Name","maxpool3d_5","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    transposedConv3dLayer([3 3 3],128,"Name","transposed-conv3d","Cropping","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    transposedConv3dLayer([3 3 3],64,"Name","transposed-conv3d_1","Cropping","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    transposedConv3dLayer([3 3 3],32,"Name","transposed-conv3d_2","Cropping","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    transposedConv3dLayer([3 3 3],16,"Name","transposed-conv3d_3","Cropping","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_9")
    convolution3dLayer([1 1 1],1,"Name","conv3d","WeightsInitializer","he")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"relu_1","concat/in1");
lgraph = connectLayers(lgraph,"relu_2","concat/in2");
lgraph = connectLayers(lgraph,"concat","conv3d_3");
lgraph = connectLayers(lgraph,"concat","addition_3/in2");
lgraph = connectLayers(lgraph,"relu_3","conv3d_4");
lgraph = connectLayers(lgraph,"relu_3","addition_2/in2");
lgraph = connectLayers(lgraph,"relu_4","conv3d_5");
lgraph = connectLayers(lgraph,"relu_4","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_6","addition_1/in1");
lgraph = connectLayers(lgraph,"relu_7","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_8","addition_3/in1");

net = dlnetwork(lgraph);

end

