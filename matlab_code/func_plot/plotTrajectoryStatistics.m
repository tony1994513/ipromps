function [q_test_mu, q_test_var, q_test_corr, plotHandles, fHan_rep] = ...
       plotTrajectoryStatistics(data, time, color)

    
    [q_test_mu, q_test_var, q_test_corr] = analyzeTrajectories(data);
    
    usePlotHandles = exist('plotHandles', 'var');
  
    for j=1:size(q_test_mu,1) / 2

        hold all

        if ( ~iscell(color))
            color = {[color, '-']};
        end
        
        %> why do they compute mean_ this way???
        mean_ = q_test_mu(2 * j - 1 + offset, :);
        if 0% sum( mean_ - q_test_mu(1,:) ) ~=0
            error('means do not match');
        end
        %> plot only the variance (no covariance)
        twoStd_ = 2*sqrt(q_test_var(2 * j - 1 + offset, 2 * j - 1 + offset,:));
        twoStdb = 2*sqrt(   q_test_var(1, 1, :)  );
        
        if 0% sum(twoStd_ - twoStdb) ~= 0
             error('vars do not match');
        end
        
        fHan_rep(j) = shadedErrorBar(time, mean_ , twoStd_ , ...
                                     {color{:}, 'LineWidth', 2'}, 1);

    end
   
end

function [q_test_mu, q_test_var, q_test_correl] = analyzeTrajectories(data)

    q_test_mu = zeros(size(data.q{1},2), length(data.q{1}));
    q_test_var = zeros(size(data.q{1},2),size(data.q{1},2), length(data.q{1})); 
    q_test_correl = zeros(size(data.q{1},2) * 2,size(data.q{1},2) * 2, length(data.q{1}));
    
    %> going through all time steps
    for i = 1:length(data.q{1})
        
        q_test = zeros(  length(data.q)   ,  size(data.q{1},2)   );
        
        % for each time step get the value of each observation
        for j = 1:length(data.q)
            q_test(j,:) = data.q{j}(i,:);
        end
        
        q_test_mu(:,i) = mean(q_test);
        
        q_test_var(:,:,i) = cov(q_test);
        
        if (i > 1)
           q_test_correl(:, :, i) =  cov([q_old q_test]);         
        end
        q_old = q_test;
    end
    
    
end