function [mapMat,map] = generateMap()  
% 输出：mapMat为（32*32*32）0-1地图矩阵，map为对应的占据栅格地图
%% 初始化参数
    str = initParam("map");
    mapSize = str.mapSize;
    obsNum = str.obsNum;
    obsMaxRange = str.obsMaxRange;
%% 初始化空白地图矩阵
    mapMat = 0*ones(mapSize);
%% 生成障碍
    % 离原点最近的顶点索引
    obsIndex1 = [randi([1,mapSize(1)-obsMaxRange],obsNum,1), ...
                 randi([1,mapSize(2)-obsMaxRange],obsNum,1), ...
                 randi([1,mapSize(3)-obsMaxRange],obsNum,1)]; 
    % 离原点最远的顶点索引
    obsIndex2 = obsIndex1+randi([1,obsMaxRange],obsNum,3);         
    % 填充障碍点
    for i = 1:obsNum                            
    mapMat(obsIndex1(i,1):obsIndex2(i,1), ...
        obsIndex1(i,2):obsIndex2(i,2), ...
        obsIndex1(i,3):obsIndex2(i,3)) = 1;
    end
%% 生成障碍点云
    [x,y,z] = ind2sub(size(mapMat),find(mapMat==1));    % 取障碍点3D索引
    xyzpoints = [x,y,z];
    ptCloud = pointCloud(xyzpoints);                    % 生成点云数据
%% 生成栅格地图
    map = occupancyMap3D(1);                    % 生成占据栅格地图
    pose = [0 0 0 1 0 0 0];                     % 位姿[x y z qw qx qy qz]
    maxRange = sqrt(mapSize*mapSize');          % 传感器半径 
    insertPointCloud(map,pose,ptCloud,maxRange)
end