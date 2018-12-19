function [kf_perJoint] = mainloop(h, data, param, observed, type)

    style1 = struct('Color', 'g', 'LineStyle', 'none', 'LineWidth', 0.7, 'Marker' , 'o',...
                   'MarkerFaceColor','auto', 'MarkerSize', 7);

    style2 = struct('Color', 'r', 'LineStyle', 'none', 'LineWidth', 0.7, 'Marker' , 'o',...
                   'MarkerFaceColor','auto', 'MarkerSize', 5);
               
    styleObs  = struct('Color', 'b', 'LineStyle', 'none', 'LineWidth', 0.2, 'Marker' , 'o',...
                   'MarkerFaceColor','b', 'MarkerSize', 10);
               

               
    % covariance between states
    P_old = data.q_cov; % all states are correlated

    % First guess is the mean
    x_old = data.q_mean';

    
    % Recursive conditioning: one sample each time
    % Observation matrix
    %  z = Hx + R_obs    
    sigma_obs        = observed.stdev;
    R_obs            = (sigma_obs^2)*eye( param.nTotalJoints ); 


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
        [H]   = formatObservation(k, param);

        % You could use some absurd numbers for non observed states
        % as they will be canceled by the Kalman gain anyway
        % z   = [ 1e20 ; observed.q(observed.joint, k);  1e20;  1e20];
        z = observed.q(:, k);


        [x_new, P_new] = KF_recursion(x_old, P_old, H, z, R_obs);
        x_old = x_new;
        P_old = P_new;

        % separate the huge full matrices for each joint (most
        % useful to plot, but useless for KF algorithm)
        kf_perJoint = perJoint(x_new, P_new, param);

        h = plotIterative(h, kf_perJoint, 'r');

        fprintf('%g  observations left\n', aux.nObsTotal-aux.counter);

        pause(0.1);
        aux.counter = aux.counter+1;
    end
    
    % =========================================


                                      
    
                     

end


