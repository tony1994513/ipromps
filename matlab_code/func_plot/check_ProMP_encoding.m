function [] = check_ProMP_encoding(pmp, data)
% Compare the ProMP with mean and variance from raw data


    % This will plot the mean and variance of trajectories based on the
    % ProMP
    [h hdot] = plot_ProMP(pmp);

    % Plot the mean and variance of the data
    plot_rawstatistics(data, pmp.nJoints, h, hdot);
    
   
    
end

function [  ] = plot_rawstatistics(data, nJoints, h, hdot)

    for j = 1:nJoints
        figure(h(j));
        plot(data(j).q_mean + 0*sqrt( data(j).q_var )', 'r', 'LineWidth', 2);
        plot(data(j).q_mean + 2*sqrt( data(j).q_var )', 'r--', 'LineWidth', 2);
        plot(data(j).q_mean - 2*sqrt( data(j).q_var )', 'r--', 'LineWidth', 2);
        figure(hdot(j));
        plot(data(j).qdot_mean + 0*sqrt( data(j).qdot_var )', 'r--', 'LineWidth', 2);
        plot(data(j).qdot_mean + 2*sqrt( data(j).qdot_var )', 'r--', 'LineWidth', 2);
        plot(data(j).qdot_mean - 2*sqrt( data(j).qdot_var )', 'r--', 'LineWidth', 2);
    end
    
end


function [h hdot] = plot_ProMP(pmp)

    % computing mean and variance based on weights and features of the
    % ProMP
    for j = 1:pmp.nJoints
        
        in = pmp.w.index{j};
  
        meanw = pmp.w.mean_full(in);
        covw  = pmp.w.cov_full(in,in);

        q_mean  = pmp.basis.Gn*meanw;
        q_Sigma = pmp.basis.Gn*covw*pmp.basis.Gn';
        q_Sigma_ii = diag(q_Sigma);
        
        qdot_mean  = pmp.basis.Gndot*meanw;
        qdot_Sigma = pmp.basis.Gndot*covw*pmp.basis.Gndot';
        qdot_Sigma_ii = diag(qdot_Sigma);
        
        
        h(j) = figurew(['joint_' num2str(j)]) ;
        plot(q_mean +  0*sqrt(q_Sigma_ii), SBLUE(3));
        plot(q_mean +  2*sqrt(q_Sigma_ii), 'b--', 'LineWidth', 2);
        plot(q_mean -  2*sqrt(q_Sigma_ii), 'b--', 'LineWidth', 2);
        
        hdot(j) = figurew(['vel_' num2str(j)]) ;
        plot(qdot_mean +  0*sqrt(qdot_Sigma_ii), SBLUE(3));
        plot(qdot_mean +  2*sqrt(qdot_Sigma_ii), 'b--', 'LineWidth', 2);
        plot(qdot_mean -  2*sqrt(qdot_Sigma_ii), 'b--', 'LineWidth', 2);
        
    end
    
end


function [h hdot] = plot_ProMP_orig(pmp)

    % computing mean and variance based on weights and features of the
    % ProMP
    for j = 1:pmp.nJoints
        
        in = pmp.w.index{j};
         
        db.x.mean  = [];
        db.x.var   = [];
        db.xd.mean = [];
        db.xd.var = [];
    
        for i = 1:(pmp.nTraj-1)

            timePoint = i*pmp.phase.dt;
            [mu_x, ~, Sigma_t] = ...
                              get_distributions_at_time_T1(...
                                               pmp.w.mean_full(in), ...
                                               pmp.w.cov_full(in,in), pmp.basis, pmp.phase.dt, timePoint);

            db.x.mean  = [db.x.mean   mu_x(1)];
            db.xd.mean = [db.xd.mean  mu_x(2)];            
            db.x.var   = [db.x.var    Sigma_t(1,1)];        
            db.xd.var  = [db.xd.var   Sigma_t(2,2)];

        end

        h(j) = figurew(['joint_' num2str(j)]) ;
        plot(db.x.mean +  0*sqrt(db.x.var), SBLUE(3));
        plot(db.x.mean +  2*sqrt(db.x.var), 'g--', 'LineWidth', 2);
        plot(db.x.mean -  2*sqrt(db.x.var), 'g--', 'LineWidth', 2);
        
        hdot(j) = figurew(['vel_' num2str(j)]) ;
        plot(db.xd.mean +  0*sqrt(db.xd.var), SBLUE(3));
        plot(db.xd.mean +  2*sqrt(db.xd.var), 'g--', 'LineWidth', 2);
        plot(db.xd.mean -  2*sqrt(db.xd.var), 'g--', 'LineWidth', 2);
        
        % plot(diff(db.x.mean)/0.01, 'go-');
        
        
    end
    
end









