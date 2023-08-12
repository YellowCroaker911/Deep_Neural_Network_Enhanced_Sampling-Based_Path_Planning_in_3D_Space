clc
clear
close all

%% 模型准备
% generateData();
% generateNetModel() % 通过matlab生成网络模型
% 运行main.py，通过python生成网络模型

%% 测试
clc
clear
close all

str = initParam("main");
example = str.example;
threshold = str.threshold;
iterMax = str.iterMax;
step = str.step;

% matlab得到的网络模型
% [X1,Y,startPos,goalPos] =  getExample();
% python得到的网络模型
[X1,Y,startPos,goalPos] = getExamplePy(example);
% promissing三维矩阵转点云
map = mat2omap(X1);
promisingPTC = mat2ptc(Y,threshold);
% 调用RRTstar
[path,nodeList] = RRTstar(map,size(X1),startPos,goalPos,iterMax,step,promisingPTC);
if ~isempty(path)
    visualizeMap(X1,"mat","path",true);
    drawMark(path,"line",true);
end
