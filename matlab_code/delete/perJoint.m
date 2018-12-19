
function kf = perJoint(xFull, Pfull, param)

    
    nTraj = param.nTraj;
    
    for j=1:param.nTotalJoints
        ind = [(j-1)*nTraj+1:1:j*nTraj];
        kf(j).q    = xFull(ind);
            P_ii_temp = diag(Pfull);
        kf(j).P_ii = P_ii_temp(ind);
    end
    
    
    
    