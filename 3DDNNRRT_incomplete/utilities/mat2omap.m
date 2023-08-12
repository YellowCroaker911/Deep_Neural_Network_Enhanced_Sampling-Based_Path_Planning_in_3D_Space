function omap = mat2omap(mapMat)
% 三维0-1矩阵转占据栅格地图
    ptc = mat2ptc(mapMat,0.5);
    omap = occupancyMap3D(1);
    pose = [0 0 0 1 0 0 0];
    maxRange = sqrt(size(mapMat)*size(mapMat)'); 
    insertPointCloud(omap,pose,ptc,maxRange)
end

