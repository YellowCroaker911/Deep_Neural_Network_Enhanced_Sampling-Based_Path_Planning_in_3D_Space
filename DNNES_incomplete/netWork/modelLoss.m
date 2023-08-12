function [loss,gradients,state] = modelLoss(net,X1,X2,T)
% 定义损失函数，这里crossentropy应该不适用于求二值交叉熵
    [Y,state] = forward(net,X1,X2);
    loss = crossentropy(Y,T);
    gradients = dlgradient(loss,net.Learnables);
end

