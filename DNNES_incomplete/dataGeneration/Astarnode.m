classdef Astarnode
    properties
        pos                 % 节点位置
        costG               % 代价G
        costH               % 代价H （代价F=G+H）    
        faIndex             % 父节点在closeList中的下标
        curIndex            % 当前节点在closeList中的下标
    end
    methods
        function obj = Astarnode(pos,costG,costH,faIndex,curIndex)
            obj.pos = pos;
            obj.costG = costG;
            obj.costH = costH;
            obj.faIndex = faIndex;
            obj.curIndex = curIndex;
        end
    end
end

