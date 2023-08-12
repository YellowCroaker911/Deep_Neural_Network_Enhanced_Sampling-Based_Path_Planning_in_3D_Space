function [X1,Y,startPos,goalPos] = getExample(example)
% 输入：example为测试样本下标
% 输出：X1为障碍地图0-1矩阵，Y为promising region三维矩阵（未设定阈值）
    load("net.mat");
    % analyzeNetwork(net)
    
    ds = loadData("trainData.mat");
    % ds = loadData("testData.mat");
    sg = load("trainStartAndGoal.mat");
    % sg = load("testStartAndGoal.mat");
    
    mbq = minibatchqueue(ds,...
    MiniBatchSize=64, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSSCB");
    
    % 取下一批
    [X1,X2,T] = next(mbq);
    Y = predict(net,X1,X2);
    X1 = dlarray2mat(X1); 
    X2 = dlarray2mat(X2); 
    T = dlarray2mat(T); 
    Y = dlarray2mat(Y);
    
    % 取example样本
    X1 = X1(:,:,:,1,example);
    X2 = X2(:,:,:,1,example);
    T = T(:,:,:,1,example);
    Y = Y(:,:,:,1,example);
    
    % 可视化
    % 输出：X1为障碍地图0-1矩阵，X2为起始点与终点的状态地图0-1矩阵
    %      T为包含由A*生成的全局最优路径地图的0-1矩阵（已拓宽）
    %      Y为promising region三维矩阵（未设定阈值）
    visualizeMap(X1,"mat","map",true);
    visualizeMap(X2,"mat","path",true);
    visualizeMap(T,"mat","path",true);
    visualizeMap(Y,"mat","netTest",true);

    % 获取起点和终点
    sg = sg.startAndGoal;
    startPos = cell2mat(sg(example,1));
    goalPos = cell2mat(sg(example,2));

end
    
function [X1,X2,T] = preprocessMiniBatch(dataX1,dataX2,dataT)
% 第5维为batch维，这里是将多个数据合并成一批
    % Preprocess predictors.
    X1 = cat(5,dataX1{1:end});
    X2 = cat(5,dataX2{1:end});
    % Extract label data from cell and concatenate.
    T = cat(5,dataT{1:end});
end

function A = dlarray2mat(A)
% 将A由dlarray类型转为mat类型后才能传进python以及可视化
    A = gpuArray(A);
    A = gather(A);
    A = extractdata(A);
end


