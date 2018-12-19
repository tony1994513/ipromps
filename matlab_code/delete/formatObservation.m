function [H] = formatObservation(k, param)
% Generic implementation of observation matrix when features are simply
% unity
    
    nStates = param.nTotalJoints;
    nTraj   = param.nTraj;
    
    % create the zero matrices
    for j=1:nStates
       H{j} = zeros(1, nTraj);
    end

    % fill the matrices with ONEs at the index where the observation occurs
    for s = param.observedJointPos
        H{s}(:,k) = 1;
    end
    
    % create the block diagonal matrix
    H = blkdiag( H{:} );
    
    
end
    
    
