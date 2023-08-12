function visualizeMap(map,type,thresholdType,enable)
% 输入：map为地图数据，type为输入类型（包括cell,mat,mark(即点云的Location）,omap)
%     thresholdType为阈值类型，（包括map，path，netTest，输出栅格地图的栅格集合反映一定阈值内的数据元素）
%     enable为使能，控制是否执行可视化
% 输出：显示占据栅格地图
%     对于障碍地图输入，反映障碍位置
%     对于起始点和终点状态地图输入，反映起始点和终点位置
%     对于A*膨胀路径地图输入，反映A*膨胀路径
%     对于将网络输出作为输入，反映promising region
    if ~enable
        return
    end
    figure();
    visualizeMap_(map,type,nan,thresholdType)
end

function visualizeMap_(map,type,mapSize,thresholdType)
    switch type
        case "cell"
            mat = cell2mat(map);
            visualizeMap_(mat,"mat",size(mat),thresholdType)
        case "mat"
            str = initParam("main");
            threshold = str.threshold;
            if thresholdType == "map"
                [x,y,z] = ind2sub(size(map),find(map>0.5));
            elseif thresholdType == "path"
                [x,y,z] = ind2sub(size(map),find(map>0.5));
            elseif thresholdType == "netTest"
                [x,y,z] = ind2sub(size(map),find(map>threshold));
            end
            xyzpoints = [x,y,z];
            visualizeMap_(xyzpoints,"mark",size(map),nan)
        case "mark"
            ptCloud = pointCloud(map);           % 生成点云数据
            omap = occupancyMap3D(1);            % 生成占据栅格地图
            pose = [0 0 0 1 0 0 0];              % 位姿[x y z qw qx qy qz]
            maxRange = sqrt(mapSize*mapSize');   % 传感器半径 
            insertPointCloud(omap,pose,ptCloud,maxRange)
            visualizeMap_(omap,"omap",nan,nan)
        case "omap"
            show(map); 
            axis equal;
    otherwise
            error('The type name does not exist!');
    end
end
