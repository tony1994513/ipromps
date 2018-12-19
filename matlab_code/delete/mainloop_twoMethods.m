function [kf_perJoint] = mainloop_twoMethods(h, pmp, data, param, observed)

%% styles
    style1 = struct('Color', 'g', 'LineStyle', 'none', 'LineWidth', 0.7, 'Marker' , 'o',...
                   'MarkerFaceColor','auto', 'MarkerSize', 7);

    style2 = struct('Color', 'r', 'LineStyle', 'none', 'LineWidth', 0.7, 'Marker' , 'o',...
                   'MarkerFaceColor','auto', 'MarkerSize', 5);
               
    styleObs  = struct('Color', 'b', 'LineStyle', 'none', 'LineWidth', 0.2, 'Marker' , 'o',...
                   'MarkerFaceColor','b', 'MarkerSize', 10);
               
%% main code

    % covariance between states
    P_old = data.q_cov; % all states are correlated

    % First guess is the mean
    x_old = data.q_mean';
    
    
    P_oldpmp = pmp.w.cov_full;
    x_oldpmp = pmp.w.mean_full;
    
    
    
    % Recursive conditioning: one sample each time
    % Observation matrix
    %  z = Hx + R_obs    
    sigma_obs        = observed.stdev;
    R_obs            = (sigma_obs^2)*eye( param.nTotalJoints );

    R_obs_pmp            = (sigma_obs^2)*eye( 2*param.nTotalJoints );
    
    % Recursive processing =====================
    aux.nObsTotal = length(observed.index);
    aux.counter = 1;
    for k = observed.index

        % plot the observation at the current step
        for jj = observed.joint
            figure( h(jj).fig ); 
            plot( k, observed.q(jj, k), styleObs); 
        end

        % create observation matrix. It has to address all the S states
        [H]      = formatObservation(k, param);
        
        % create observation matrix. Includes the basis for velocities will
        % also be updated
        [Hpmp]   = formatObservationPMP(k, pmp, observed.joint);        
        

        % You could use some absurd numbers for non observed states
        % as they will be canceled by the Kalman gain anyway
        % z   = [ 1e20 ; observed.q(observed.joint, k);  1e20;  1e20];
        z = observed.q(:, k);
        
        zpmp = [];
        for j=1:pmp.nJoints
            q_qdot = [observed.q(j, k); observed.qdot(j, k)];
            zpmp = [zpmp; q_qdot];
        end

        % Original method without features
        [x_new, P_new] = KF_recursion(x_old, P_old, H, z, R_obs);
        x_old = x_new;
        P_old = P_new;
        
        % PMP method
        [x_newpmp, P_newpmp] = KF_recursion(x_oldpmp, P_oldpmp, Hpmp, zpmp, R_obs_pmp);
        x_oldpmp = x_newpmp;
        P_oldpmp = P_newpmp;

        % separate the huge full matrices for each joint (most
        % useful to plot, but useless for KF algorithm)
        kf_perJoint     = perJoint(x_new, P_new, param);
        kf_perJoint_pmp = perJoint_pmp(x_newpmp, P_newpmp, pmp);
        

        h = plotIterative(h, kf_perJoint, 'r');
        
        for j=1:4
           try
               delete(hTemp(j,:))
           end
           figure(h(j).fig)
           hTemp(j,1) = plot(kf_perJoint_pmp(j).q_mean, 'b-');
           hTemp(j,2) = plot(kf_perJoint_pmp(j).q_mean + 2*sqrt(kf_perJoint_pmp(j).q_Sigma_ii), 'b-');
           hTemp(j,3) = plot(kf_perJoint_pmp(j).q_mean - 2*sqrt(kf_perJoint_pmp(j).q_Sigma_ii), 'b-');
        end
        
pause        

        fprintf('%g  observations left\n', aux.nObsTotal-aux.counter);

        pause(0.1);
        aux.counter = aux.counter+1;
    end
    
    % =========================================


                                      
    
                     

end


