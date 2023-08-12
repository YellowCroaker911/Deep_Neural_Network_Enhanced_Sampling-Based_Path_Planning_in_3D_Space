function generateData()
% 输出：将数据集以及相应的起始点和终点位置保存到项目根目录
% dataCell1-2列为网络模型的2个输入，3列为网络模型输出
    %% 初始化参数
    str = initParam("data");
    dataNum = str.dataNum;
    mapNum = str.mapNum;
    iterMax = str.iterMax;
    inflateRange = str.inflateRange;
    visualization = str.visualization;
    for set = 1:2
        %% 初始化空数据集元胞
        dataCell = cell(dataNum,3);
        startAndGoal = cell(dataNum,2);
        %% 生成地图
        % 本应由点云转二值矩阵，这里方便起见直接将生成点云的二值矩阵放入数据集
        % 栅格地图则作为A*输入
        mapMats = cell(mapNum,1);
        maps = repelem(occupancyMap3D(),mapNum,1);
        for i = 1:mapNum
            [mapMat,map] = generateMap();
            mapSize = size(mapMat);
            mapMats(i) = mat2cell(mapMat,mapSize(1),mapSize(2),mapSize(3));
            maps(i) = map;
        end
        %% 生成数据
        delta = dataNum/mapNum;
        for i = 1:mapNum
            for j = 1:delta
                idx = (i-1)*delta+j;
                disp("Current Iterations:"+idx);
                mapMat = cell2mat(mapMats(i));
                map = maps(i);
                while true
                    startPos = sample(mapMats(i),inflateRange);
                    goalPos = sample(mapMats(i),inflateRange);
                    
                    % ↓---可视化---↓
                    visualizeMap(mapMat,"mat","map",visualization);
                    drawMark([startPos;goalPos],"point",visualization);
    
                    path = Astar(map,size(mapMat),startPos,goalPos,iterMax,inflateRange);
                   
                    drawMark(path,"line",visualization);
                    if visualization
                        disp("length of path:"+length(path));
                        pause(5);
                        close all;
                    end
                    % ↑---可视化---↑
    
                    if isempty(path)
                        continue;
                    else
                        stateMat = 0*ones(mapSize);
                        stateMat = fill(stateMat,dilate(mapMat,startPos,inflateRange),1);
                        stateMat = fill(stateMat,dilate(mapMat,goalPos,inflateRange),1);
                        pathMat = 0*ones(mapSize);
                        pathMat = fill(pathMat,dilate(mapMat,path,inflateRange),1);
                        dataCell(idx,1) = mapMats(i);
                        dataCell(idx,2) = mat2cell(stateMat,mapSize(1),mapSize(2),mapSize(3));
                        dataCell(idx,3) = mat2cell(pathMat,mapSize(1),mapSize(2),mapSize(3));
                        startAndGoal(idx,1) = mat2cell(startPos,1,3);
                        startAndGoal(idx,2) = mat2cell(goalPos,1,3);
                        break
                    end
                end
                
            end
        end
        if set == 1
            save("trainData.mat","dataCell");
            save("trainStartAndGoal.mat","startAndGoal");
        else
            save("testData.mat","dataCell");
            save("testStartAndGoal.mat","startAndGoal");
        end
    end
end
%% 采样无障碍点
function freePos = sample(mapMatCell,range)
    mapMat = cell2mat(mapMatCell);
    mapSize = size(mapMat);
    while true
        point = [randi([1,mapSize(1)]), ...
                 randi([1,mapSize(2)]), ...
                 randi([1,mapSize(3)])];
        freepoints = detect(mapMat,point,range);
        if length(freepoints) ~= (2*range+1)^3        
            continue
        else
            freePos = point;
            break
        end
    end 
end
%% 障碍与边界检测
function freePoints = detect(mapMat,point,range)
% 返回-range:range正方体范围内的未越界无障碍点
% （采用这个形式是为了方便调用函数获得输入位置周边的无障碍位置从而进行膨胀拓宽）
    mapSize = size(mapMat);
    freePoints = [];
    
    for i = -range:range
        for j = -range:range
            for k = -range:range

                posOffset = [i,j,k];
                pos = point + posOffset;

                if pos(1) < 1 || pos(1) > mapSize(1) || ...
                   pos(2) < 1 || pos(2) > mapSize(2) || ...
                   pos(3) < 1 || pos(3) > mapSize(3)
                   continue;
                end

                if mapMat(pos(1),pos(2),pos(3)) == 1
                   continue;
                end

                freePoints = [freePoints;pos];

            end
        end
    end
end
%% 拓宽膨胀标记点
function marks = dilate(mapMat,marks,range)
% 将marks中所有点在-range:range正方体范围内的未越界无障碍点加入marks
    for l = 1:size(marks,1)
        freePoints = detect(mapMat,marks(l,:),range);
        marks = [marks;freePoints];
    end
    marks = unique(marks,"rows");
end
%% 填充标记点矩阵
function markMat = fill(markMat,marks,value)
    for i = 1:size(marks,1)
        markMat(marks(i,1),marks(i,2),marks(i,3)) = value;
    end
end