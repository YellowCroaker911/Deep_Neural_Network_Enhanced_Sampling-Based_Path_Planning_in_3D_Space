function [path,nodeList] = RRTstar(map,mapSize,startPos,goalPos,iterMax,step,promisingRegion)
% 输入：map为占据栅格地图，mapSize为地图尺寸大小，startPos为起点，goalPos为目标点
%      iterMax为最大迭代次数，step为迭代步长，promisingRegion为期望区域
    str = initParam("RRTstar");
    visualization = str.visualization;

    startNode = RRTnode(startPos,nan,1); 
    goalNode = RRTnode(goalPos,nan,nan);
    nodeList = startNode;
    hadfoundFeisiblePath = false;
    path = [];

    pathLineHandles = [];
    treeLineHandles = [];

    for i=1:iterMax
        disp("Current Iterations:"+i);
        samplePoint = sample_point(mapSize,goalNode,promisingRegion,hadfoundFeisiblePath);
        nearestNode = find_nearest_node(nodeList,samplePoint);
        newNode = generate_new_node(step,samplePoint,nearestNode);
        nearNodes = find_near_nodes(step*2,nodeList,newNode);
        sortedNearNodes = sort_near_nodes_bycost(nodeList,newNode,nearNodes);
        fatherNode = pick_father_node(map,newNode,sortedNearNodes);
        if isempty(fatherNode) == false
            [nodeList,newNode] = add_new_node(nodeList,newNode,fatherNode);
            nodeList = rewrite(map,nodeList,newNode,sortedNearNodes);
            if hadfoundFeisiblePath == false
                isfoundFeisiblePath = foundedpath_judge(map,goalNode,step,newNode);
            end
            if isfoundFeisiblePath
                if hadfoundFeisiblePath == false
                    [nodeList,goalNode] = add_new_node(nodeList,goalNode,newNode);
                    hadfoundFeisiblePath = true;
                else
                    path = get_path(nodeList,goalNode);
                    pathLineHandles = show_path(path,pathLineHandles,visualization);
                end
            end
        end
        treeLineHandles = show_nodeList(nodeList,treeLineHandles,visualization);
    end
end
%% 随机采样（这里的采样机制根据原论文）
function samplePoint = sample_point(mapSize,goalNode,promisingRegion,hadfoundFeisiblePath)
    mu1 = 0.9; mu2 = 0.5;
    mu = mu1;
    if hadfoundFeisiblePath
        mu = mu2;
    end
    if rand() < mu                              % 非均匀采样
        idx = randi([1,promisingRegion.Count]);
        samplePoint = promisingRegion.Location(idx,:);
    else                                        % 均匀采样
        samplePoint = rand(1,3).*mapSize;
        % % % 标准RRT/RRT*算法中有分配一定概率取目标点作为采样点的机制，即：
        % % if rand() < 0.5
        % %     samplePoint = rand(1,3).*mapSize;
        % % else
        % %     samplePoint = goalNode.pos;
        % % end
    end
end
%% 寻找最近节点
function nearestNode = find_nearest_node(nodeList,samplePoint)
    delta = zeros(length(nodeList),3);
    for i = 1:length(nodeList)
        delta(i,:) = nodeList(i).pos-samplePoint;
    end
    dis2 = sum(delta.*delta,2);
    [~,index] = min(dis2(:));
    nearestNode = nodeList(index);
end
%% 生成新节点
function newNode = generate_new_node(step,samplePoint,nearestNode)
    delta = samplePoint-nearestNode.pos;
    radius = sqrt(sum(delta.*delta));
    fai = atan2(delta(2),delta(1));
    theta = acos(delta(3)/radius);
    offset = [step*sin(theta)*cos(fai), ...
              step*sin(theta)*sin(fai), ...
              step*cos(theta)];
    newPos = nearestNode.pos + offset;
    newNode = RRTnode(newPos,nan,nan);
end
%% 寻找近节点集
function nearNodes = find_near_nodes(radius,nodeList,newNode)
    dis = zeros(length(nodeList),1);
    for i = 1:length(nodeList)
        dis(i,:) = calculateDis(nodeList(i),newNode);
    end
    nearNodes = nodeList(dis(:) < radius);
end
%% 根据到起始节点与新节点距离之和排序近节点集
function sortedNearNodes = sort_near_nodes_bycost(nodeList,newNode,nearNodes)
    dis2start = zeros(length(nearNodes),1);
    dis2new = zeros(length(nearNodes),1);
    for i=1:length(nearNodes)
        dis2new(i,1) = calculateDis(nearNodes(i),newNode);
        dis2start(i,1) = calculateDis2start(nodeList,nearNodes(i));
    end
    cost = dis2start+dis2new;
    [~,idx] = sort(cost);
    sortedNearNodes = nearNodes(idx);
end
%% 筛选父节点（含碰撞检测）
function fatherNode = pick_father_node(map,newNode,sortedNearNodes)
    for i=1:length(sortedNearNodes)
        maxrange = calculateDis(newNode,sortedNearNodes(i));
        if maxrange == 0
            continue
        end
        iscollided = collision_detection(map,maxrange,sortedNearNodes(i),newNode);
        if iscollided == false
            fatherNode = sortedNearNodes(i);
            return
        end
    end
    fatherNode = [];
end
%% 碰撞检测
function iscollided = collision_detection(map,maxrange,node1,node2)
    sensorPose = [node1.pos,1,0,0,0];
    directions = node2.pos-node1.pos;
    [~,isOccupied] = ...
        rayIntersection(map,sensorPose,directions,maxrange);
    if isOccupied == 1
        iscollided = true;
    else
        iscollided = false;
    end
    % 可视化（若要启用可视化，可能需要修改一下代码才会有好的效果，即只有在合适的调用函数下才进行可视化）
%     show_collision_detection(sensorPose,intersectionPts,isOccupied);
end
%% 加入新节点
function [updatedNodeList,updatedNewNode] = add_new_node(nodeList,newNode,fatherNode)
    newNode.faIndex = fatherNode.curIndex;
    newNode.curIndex = length(nodeList)+1;
    updatedNewNode = newNode;
    updatedNodeList = [nodeList,newNode];
end
%% 改写近节点集父节点
function nodeList = rewrite(map,nodeList,newNode,NearNodes)
    disNew2start = calculateDis2start(nodeList,newNode);
    for i=1:length(NearNodes)
        if newNode.faIndex == NearNodes(i).curIndex
            continue
        end
        dis2new = calculateDis(NearNodes(i),newNode);
        if dis2new == 0
            continue
        end
        iscollided = collision_detection(map,dis2new,NearNodes(i),newNode);
        if iscollided
            continue
        end
        dis2start = calculateDis2start(nodeList,NearNodes(i));
        if dis2start > dis2new + disNew2start
            nodeList(NearNodes(i).curIndex).faIndex = newNode.curIndex;
        end
    end
end
%% 找到可行路径判断
function isfoundFeisiblePath = foundedpath_judge(map,goalNode,step,newNode)
    isfoundFeisiblePath = false;
    delta = goalNode.pos-newNode.pos;
    dis = sqrt(sum(delta.*delta));
    if dis < step
        iscollided = collision_detection(map,step,newNode,goalNode);
        if iscollided == false
            isfoundFeisiblePath = true;
        end
    end
end
%% 获取路径
function path = get_path(nodeList,goalNode)
    path = goalNode.pos;
    idx = goalNode.faIndex;
    while isnan(idx) == false
        path = [nodeList(idx).pos;path];
        idx = nodeList(idx).faIndex;
    end
end
%% 通用操作
%% 计算两节点之间的距离
function dis = calculateDis(node1,node2)
    delta = node1.pos-node2.pos;
    dis = sqrt(sum(delta.*delta));
end
%% 计算到起始节点的距离
function dis2start = calculateDis2start(nodeList,node)
    idx = node.curIndex;
    dis2start = 0;
    while isnan(nodeList(idx).faIndex) == false
        disTemp = calculateDis(nodeList(nodeList(idx).faIndex),nodeList(idx));
        dis2start = dis2start + disTemp; 
        idx = nodeList(idx).faIndex;
    end
end
%% 可视化
%% 展示可行路径
function newLineHandles = show_path(path,lineHandles,visualization)
    if ~visualization
        newLineHandles = [];
        return;
    end
    hold on;
    for i=1:length(lineHandles)
        delete(lineHandles(i))
    end
    newLineHandles = zeros(length(path)-1);
    for i=1:length(path)-1
        lineHandle = line(path(i:i+1,1),path(i:i+1,2),path(i:i+1,3),color="green",LineWidth=3);
        newLineHandles(i) = lineHandle;
    end
    drawnow();
end
%% 展示树
function newLineHandles = show_nodeList(nodeList,lineHandles,visualization)
    if ~visualization
        newLineHandles = [];
        return;
    end
    hold on;
    for i=1:length(lineHandles)
        delete(lineHandles(i))
    end
    newLineHandles = zeros(length(nodeList)-1);
    for i=2:length(nodeList)
        x = [nodeList(i).pos(1);nodeList(nodeList(i).faIndex).pos(1)];
        y = [nodeList(i).pos(2);nodeList(nodeList(i).faIndex).pos(2)];
        z = [nodeList(i).pos(3);nodeList(nodeList(i).faIndex).pos(3)];
        lineHandle = line(x,y,z,color="blue",LineWidth=1);
        newLineHandles(i-1) = lineHandle;
    end
    drawnow();
end
%% 展示碰撞检测
function show_collision_detection(sensorPose,intersectionPts,isOccupied)
    hold on;
    plotTransforms(sensorPose(1:3),sensorPose(4:end), ...
                   "FrameSize",0.5) % Vehicle sensor pose
    for i = 1:1
        plot3([sensorPose(1),intersectionPts(i,1)],...
              [sensorPose(2),intersectionPts(i,2)],...
              [sensorPose(3),intersectionPts(i,3)],'-b') % Plot rays
        if isOccupied(i) == 1
            plot3(intersectionPts(i,1), ...
                  intersectionPts(i,2), ...
                  intersectionPts(i,3),'*r') % Intersection points
        end
        drawnow();
    end
end