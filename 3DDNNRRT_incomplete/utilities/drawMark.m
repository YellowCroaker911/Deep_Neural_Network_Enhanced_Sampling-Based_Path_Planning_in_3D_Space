function drawMark(marks,type,enable)
% 输入：marks为标记数据，type为类型（包括line和point），enable为使能，控制是否执行可视化
    if ~enable
        return
    end
    switch type
        case "line"
            hold on;
            for i = 1:size(marks,1)-1
                line(marks(i:i+1,1),marks(i:i+1,2),marks(i:i+1,3), ...
                    color="green",LineWidth=3);
            end
            drawnow();
        case "point"
            hold on;
            for i = 1:size(marks,1)
                scatter3(marks(i,1),marks(i,2),marks(i,3));
            end
            drawnow();
        otherwise
            error('The type name does not exist!');
    end
end

