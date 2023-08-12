classdef RRTnode
    properties
        pos,            % 节点位置
        faIndex,        % 父节点在nodeList中的下标
        curIndex,       % 当前节点在nodeList中的下标
    end
    methods
        function obj = RRTnode(pos,faIndex,curIndex)
            obj.pos = pos;
            obj.faIndex = faIndex;
            obj.curIndex = curIndex;
        end
    end
end

