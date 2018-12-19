function [data] = loadData( dt )


    % load abcd data here
    %load('qdemo.mat');
    
    plotTrainingData.test = 0;          % plot the test set on the top of the shadows
    plotTrainingData.test_savePlot = 0; % save the svg file
    plotTrainingData.xy   = 0;          % plot the training samples in task space
    plotTrainingData.xy_savePlot = 0;   % save the svg file
    plotTrainingData    = []; % no plot at all    
    [ab, be, cd, dc] = load_letter_learning_data(pwd, plotTrainingData);
    
    nSize = length( ab.tr.q{1}(:,1) ) ;
    
    nDemoTrain = size(ab.tr.q, 1);
    clear data
    for j=1:4
        data(j).q = []; % joint1
        data(j).qdot = []; % joint1
    end
    for d = 1:nDemoTrain
        for j=1:4
            data(j).q = [data(j).q; ab.tr.q{d}(:,2*j-1)'];
        end        
    end
    
    % compute velocity based on dt
    for j=1:4
        qdot_ = diff(data(j).q, 1, 2)/dt;
        qdot_ = [qdot_ qdot_(:,end)];
        data(j).qdot = qdot_;
    end
    
    for j=1:4
        data(j).q_mean = mean(data(j).q);
        data(j).q_cov  = cov(data(j).q);
        data(j).q_var  = diag(data(j).q_cov);
        
        data(j).qdot_mean = mean(data(j).qdot);
        data(j).qdot_cov  = cov(data(j).qdot);
        data(j).qdot_var  = diag(data(j).qdot_cov);
        
        data(j).nSize  = nSize;
    end
    

end

