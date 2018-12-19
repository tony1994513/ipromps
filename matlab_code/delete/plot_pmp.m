function [] = plot_pmp(pmp, data)
% Compare the ProMP with mean and variance from raw data

    [h hdot] = plot_ProMP(pmp);

    plot_rawstatistics(data, pmp.nJoints, h, hdot);
    
end


function [h hdot] = plot_ProMP(pmp)

    % computing mean and variance based on weights and features
    for j = 1:pmp.nJoints
        
        in = pmp.w.index{j};
         
        db.x.mean  = [];
        db.x.var   = [];
        db.xd.mean = [];
        db.xd.var = [];
    
        for i = 1:(pmp.nTraj-1)

            timePoint = i*pmp.phase.dt;
            [mu_x, ~, Sigma_t] = ...
                              get_distributions_for_time_point_T1(...
                                               pmp.w.mean_full(in), ...
                                               pmp.w.cov_full(in,in), pmp.basis, pmp.phase.dt, timePoint);

            db.x.mean  = [db.x.mean   mu_x(1)];
            db.xd.mean = [db.xd.mean  mu_x(2)];            
            db.x.var   = [db.x.var    Sigma_t(1,1)];        
            db.xd.var  = [db.xd.var   Sigma_t(2,2)];

        end
        h(j) = figurew(['joint_' num2str(j)]) ;
        plot(db.x.mean +  0*sqrt(db.x.var), SBLUE(3));
        plot(db.x.mean +  2*sqrt(db.x.var), SBLUE(3));
        plot(db.x.mean -  2*sqrt(db.x.var), SBLUE(3));
        
        hdot(j) = figurew(['vel_' num2str(j)]) ;
        plot(db.xd.mean +  0*sqrt(db.xd.var), SBLUE(3));
        plot(db.xd.mean +  2*sqrt(db.xd.var), SBLUE(3));
        plot(db.xd.mean -  2*sqrt(db.xd.var), SBLUE(3));
    end
    
end

function [  ] = plot_rawstatistics(data, nJoints, h, hdot)

    for j = 1:nJoints
        figure(h(j));
        plot(data(j).q_mean + 0*sqrt( data(j).q_var )', 'r');
        plot(data(j).q_mean + 2*sqrt( data(j).q_var )', 'r');
        plot(data(j).q_mean - 2*sqrt( data(j).q_var )', 'r');
        figure(hdot(j));
        plot(data(j).qdot_mean + 0*sqrt( data(j).qdot_var )', 'r');
        plot(data(j).qdot_mean + 2*sqrt( data(j).qdot_var )', 'r');
        plot(data(j).qdot_mean - 2*sqrt( data(j).qdot_var )', 'r');
    end
    
end