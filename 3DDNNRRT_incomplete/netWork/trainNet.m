function net = trainNet(net,ds)

    numEpochs = 100;
    miniBatchSize = 64;

    initialLearnRate = 0.000001;
    decay = 0.01;
    momentum = 0.9;

    mbq = minibatchqueue(ds,...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSSCB");
    
    velocity = [];

    numObservationsTrain = 64;  % 数据集数量
    numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
    numIterations = numEpochs * numIterationsPerEpoch;
 
    monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

    epoch = 0;
    iteration = 0;
    
    % Loop over epochs.
    while epoch < numEpochs && ~monitor.Stop
        
        epoch = epoch + 1;
    
        % Shuffle data.
        shuffle(mbq);
        
        % Loop over mini-batches.
        while hasdata(mbq) && ~monitor.Stop
    
            iteration = iteration + 1;
            
            % Read mini-batch of data.
            [X1,X2,T] = next(mbq);
            
            % Evaluate the model gradients, state, and loss using dlfeval and the
            % modelLoss function and update the network state.
            [loss,gradients,state] = dlfeval(@modelLoss,net,X1,X2,T);
            net.State = state;
            
            % Determine learning rate for time-based decay learning rate schedule.
            learnRate = initialLearnRate/(1 + decay*iteration);
            
            % Update the network parameters using the SGDM optimizer.
            [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
            
            % Update the training progress monitor.
             recordMetrics(monitor,iteration,Loss=loss);
             updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
             monitor.Progress = 100 * iteration/numIterations;
        end
    end

end

function [X1,X2,T] = preprocessMiniBatch(dataX1,dataX2,dataT)
% 第5维为batch维，这里是将多个数据合并成一批
    % Preprocess predictors.
    X1 = cat(5,dataX1{1:end});
    X2 = cat(5,dataX2{1:end});
    % Extract label data from cell and concatenate.
    T = cat(5,dataT{1:end});
end
