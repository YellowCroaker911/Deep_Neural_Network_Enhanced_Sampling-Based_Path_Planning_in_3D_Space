function str = initParam(name)
    str = struct();
    switch name
        case "map"
            % 地图大小置为原论文网络输入输出大小（不与输入输出大小一致地将会变换为一致，这里没实现）
            str.mapSize = [32,32,32];
            % 长方体障碍数量
            str.obsNum = 30;
            % 长方体障碍最大长宽高
            str.obsMaxRange = 5;
        case "data"
            % 数据集大小
            str.dataNum = 64;
            % 地图数量
            str.mapNum = 64;
            % A*算法最大迭代次数
            str.iterMax = 1024;
            % A*算法寻优路径时栅格地图的膨胀大小
            str.inflateRange = 1;
            % 可视化（显示各数据的地图与路径）
            str.visualization = false;
        case "Astar"
            % 可视化（闭集迭代）
            str.visualization = false;
        case "RRTstar"
            % 可视化（RRT树迭代）
            str.visualization = false;
        case "main"
            % 测试样本（从数据集中获得）
            str.example = 2;
            % 输出所置阈值（原论文提到输出时设置一个阈值来划分promising region）
            % 这里设置阈值后通过可视化函数：
            % visualizeMap(map,type,mapSize,thresholdType="netTest",enable)
            % 可显示栅格占据处的集合即promising region
            str.threshold = 0.2;
            % RRT*算法最大迭代次数
            str.iterMax = 1000;
            % RRT*算法步长
            str.step = 2;
        otherwise
            error('The structure name does not exist!');
    end
end