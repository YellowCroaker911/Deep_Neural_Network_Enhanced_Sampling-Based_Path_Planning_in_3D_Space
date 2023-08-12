function ptc = mat2ptc(matMap,threshold)
% 三维0-1矩阵转点云
    [x,y,z] = ind2sub(size(matMap),find(matMap>threshold));
    xyzpoints = [x,y,z];
    ptc = pointCloud(xyzpoints); 
end