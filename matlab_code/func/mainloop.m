function [joint_KF] = mainloop(h, pmp, param, observed)


%% figures and styles

    style1 = struct('Color', 'g', 'LineStyle', 'none', 'LineWidth', 0.7, 'Marker' , 'o',...
                   'MarkerFaceColor','auto', 'MarkerSize', 7);

    style2 = struct('Color', 'r', 'LineStyle', 'none', 'LineWidth', 0.7, 'Marker' , 'o',...
                   'MarkerFaceColor','auto', 'MarkerSize', 5);
               
    styleObs  = struct('Color', 'b', 'LineStyle', 'none', 'LineWidth', 0.2, 'Marker' , 'o',...
                   'MarkerFaceColor','b', 'MarkerSize', 10);
               
    % prepare the figures
    hxy(1).xy = figurew('human'); title 'Predicted human movement';
    hxy(1).dots = 0;
    set_fig_position( [0.216 0.342 0.241 0.392]   );
    xlabel 'X'; ylabel 'Y';
    hxy(2).xy = figurew('robot'); title 'Predicted robot movement';
    hxy(2).dots = 0;
    set_fig_position( [0.456 0.343 0.241 0.392]   );
    xlabel 'X'; ylabel 'Y';

%% main code

    sigma_obs        = observed.stdev;
    
    % feature: Gaussian
    P0   = pmp.w.cov_full;
    x0   = pmp.w.mean_full;
    Robs_pmp  = (sigma_obs^2)*eye( 2*param.nTotalJoints );
    
    % Recursive processing =====================
    aux.nObsTotal = length(observed.index);
    aux.counter = 1;
    
    for k = observed.index

        % plot the observation at the current step
        for jj = observed.joint
            figure( h(jj).q ); 
            plot( k, observed.q(jj, k), styleObs); 
        end

        % create observation matrix
        H0   = observation_matrix(k, pmp, observed.joint, observed.jointvel);        
        
        z0 = []; % z0 is the observed vector y*
        for j=1:pmp.nJoints
            q_qdot = [observed.q(j, k); observed.qdot(j, k)];
            z0  = [z0; q_qdot];
        end
       
        % Gaussian conditioning using Kalman filtering
        % Equation (5) and (6) of the ProMP paper (NIPS2013)
        [x1, P1] = KF_recursion(x0, P0, H0, z0, Robs_pmp);
        x0 = x1;
        P0 = P1;

        % separate the huge full matrices per each joint (most
        % useful to plot, but useless for the KF itself)
        joint_KF = perJoint_pmp(x1, P1, pmp);
        
        fprintf('%g  observations left\n', aux.nObsTotal-aux.counter);
        
        
        if 1 % plot iteratively
            h = plotTypeIterative(h, pmp.nTraj, joint_KF, 'shade');
        else
            h = plotTypeIterative(h, pmp.nTraj, joint_KF, 'edge');
        end
        
        
        if 0 % plotting: not working the way I want
             % add noise on weights
           hxy = plot2DwrapC(hxy, joint_KF, pmp.basis);
        else
           % adding noise on trajectory
           hxy = plot2DwrapB(hxy, joint_KF);
        end
        drawnow;

pause
        aux.counter = aux.counter+1;
    end
                                      
    
                     

end




function [h] = plotTypeIterative(h, nTraj, joint_KF, type)


    for j=1:4
        
        if 0 % plot lots of dots
            try delete(h(j).dot1D); end
            figure(h(j).q)                
            h(j).dot1D = plot1D(h, [1:nTraj-1], joint_KF(j).q_mean, joint_KF(j).q_Sigma_ii, 500);
        end
        
        
        switch type
            case 'shade'
                try    
                    delete(h(j).qshade.mainLine); delete(h(j).qshade.edge); delete(h(j).qshade.patch);
                    delete(h(j).qdotshade.mainLine); delete(h(j).qdotshade.edge); delete(h(j).qdotshade.patch);
                end
                figure(h(j).qdot)
                h(j).qdotshade = shadedErrorBar([1:nTraj-1], joint_KF(j).qdot_mean, 2*sqrt( joint_KF(j).qdot_Sigma_ii ) , ...
                                          {'r', 'LineWidth', 2'}, 1);     

                figure(h(j).q)
                h(j).qshade = shadedErrorBar([1:nTraj-1], joint_KF(j).q_mean, 2*sqrt( joint_KF(j).q_Sigma_ii ) , ...
                                          {'r', 'LineWidth', 2'}, 1);
            case 'edge'
               try    delete(hTemp.q(j,:));      end
               figure(h(j).q)
               hTemp.q(j,1) = plot(joint_KF(j).q_mean, 'b-');
               hTemp.q(j,2) = plot(joint_KF(j).q_mean + 2*sqrt(joint_KF(j).q_Sigma_ii), 'b-');
               hTemp.q(j,3) = plot(joint_KF(j).q_mean - 2*sqrt(joint_KF(j).q_Sigma_ii), 'b-');  
        end
        
        

    end        
    
end


