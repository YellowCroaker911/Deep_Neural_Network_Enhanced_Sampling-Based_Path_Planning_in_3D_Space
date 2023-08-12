clc
clear
close all

[mapMat,map] = generateMap();
visualizeMap(map,"omap","map",true);
path = Astar(map,[32,32,32],[1,1,1],[32,32,32],300,1);
show_path(path);

%% 展示可行路径
function show_path(path)
    hold on;
    for i=1:length(path)-1
        line(path(i:i+1,1),path(i:i+1,2),path(i:i+1,3),color="green",LineWidth=3);
    end
    drawnow();
end