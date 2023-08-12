function path = Astar(map,mapSize,startPos,goalPos,iterMax,inflateRange)
% 输入：map为占据栅格地图，mapSize为地图尺寸大小，startPos为起点，goalPos为目标点
%      iterMax为最大迭代次数，inflateRange为栅格地图膨胀大小
    str = initParam("Astar");
    visualization = str.visualization;
    
    map = copy(map);
    inflate(map,inflateRange);

    startNode = Astarnode(startPos,0,diagonal(startPos,goalPos),nan,nan);
    openList = repelem(Astarnode(nan,nan,nan,nan,nan),0,1);
    openList(1,1) = startNode;
    closeList = repelem(Astarnode(nan,nan,nan,nan,nan),0,1);
    path = [];
    
    for iter = 1:iterMax

        if isempty(openList) == false
            
            costF = zeros(length(openList),1);
            for i = 1:length(openList)
                costF(i,1) = 0.8*openList(i,1).costG+openList(i,1).costH;  
            end % 加权打破距离平衡，加速收敛

            % 模拟从优先栈中取新节点
            minCost = min(costF);
            idx = find(abs(costF(:) - minCost) <= 1E-4);  % 可能存在浮点误差
            idx = idx(end);

            if isequal(openList(idx).pos,goalPos)
                path = getPath(closeList,openList(idx));
                break
            else
                nodeTemp = openList(idx);
                openList(idx) = [];
                nodeTemp.curIndex = length(closeList)+1;
                closeList = [closeList;nodeTemp];
                drawMark(nodeTemp.pos,"point",visualization);
            end

            % 遍历邻节点
            for i = -1:1
                for j = -1:1
                    for k = -1:1

                        posOffset = [i,j,k];
                        idx = find(posOffset(:) == 0);
                        if length(idx) == 3
                            continue;
                        elseif length(idx) == 2
                            costOffsetG = 1; 
                        elseif length(idx) == 1
                            costOffsetG = sqrt(2);
                        else
                            costOffsetG = sqrt(3); 
                        end

                        pos = nodeTemp.pos + posOffset;
                        costG = nodeTemp.costG + costOffsetG;
                        costH = diagonal(pos,goalPos);
                        faIndex = nodeTemp.curIndex;
                        
                        % 越界
                        if pos(1) < 1 || pos(1) > mapSize(1) || ...
                           pos(2) < 1 || pos(2) > mapSize(2) || ...
                           pos(3) < 1 || pos(3) > mapSize(3) 
                           continue;
                        end
                        
                        % 存在障碍
                        if getOccupancy(map,pos) > 0.5 
                            continue;
                        end

                        idx = 0;
                        for l = 1:length(closeList)
                            if isequal(closeList(l).pos,pos) 
                                idx = l;
                            end
                        end
                        if idx ~= 0 % 已在closeList中
                            continue;
                        end
                        
                        idx = 0;
                        for l = 1:length(openList)
                            if isequal(openList(l).pos,pos)
                                idx = l;
                            end
                        end
                        if idx ~= 0 % 已在openList中
                            if openList(idx).costG > costG
                                openList(idx).costG = costG;
                                openList(idx).faIndex = faIndex;
                            end
                        else
                            newNode = Astarnode(pos,costG,costH,faIndex,nan);
                            openList = [openList;newNode];
                        end

                    end
                end
            end

        end
    end
end
%% 对角距离代价函数
    function cost = diagonal(pos1,pos2)
    delta = abs(pos1 - pos2);
    [sortedDelta,~] = sort(delta);
    dig1 = sortedDelta(1);
    dig2 = sortedDelta(2)-dig1;
    dig3 = sortedDelta(3)-dig2-dig1;
    cost = sqrt(3*dig1^2)+sqrt(2*dig2^2)+dig3;
end
%% 获取路径
function path = getPath(nodeList,goalNode)
    path = goalNode.pos;
    idx = goalNode.faIndex;
    while isnan(idx) == false
        path = [nodeList(idx).pos;path];
        idx = nodeList(idx).faIndex;
    end
end


