function [data, h] = plot_nStates(data)

    nSize = length(data(1).q_mean);
    
    for j=1:4
        
        if (j==1)||(j==2)
            name_aux1 = 'human';
        else
            name_aux1 = 'robot';
        end
        
        if (j==1)||(j==3)
            name_aux2 = 'x';
        else
            name_aux2 = 'y';
        end
        
        
        data(j).hq = figurew([name_aux1 '_'  name_aux2 '_' num2str(j) ]);
        Hq{j}      = shadedErrorBar([1:nSize], data(j).q_mean , 2*sqrt( data(j).q_var ) , ...
                                  {'g', 'LineWidth', 2'}, 1);
        xlabel 'state index';
        ylabel 'position value';                              

        
        data(j).hqdot =  figurew([name_aux1 '_'  name_aux2 'dot_', num2str(j)]);
        Hqdot{j}         = shadedErrorBar([1:nSize], data(j).qdot_mean , 2*sqrt( data(j).qdot_var ) , ...
                                  {'g', 'LineWidth', 2'}, 1);
        xlabel 'state index';
        ylabel 'velocity value';    
        
        
        if 0  % keep only the edges on the plot
            delete(Hq{j}.mainLine);
            delete(Hq{j}.patch);

            delete(Hqdot{j}.mainLine);
            delete(Hqdot{j}.patch); 
        end
        if 1  % keep only the patch
            delete(Hq{j}.mainLine);
            delete(Hq{j}.edge);

            delete(Hqdot{j}.mainLine);
            delete(Hqdot{j}.edge); 
        end
        grid off;
    end
    
    figure( data(1).hq  ); set_fig_position( [0.367 0.506  0.293 0.468]   );
    figure( data(2).hq  ); set_fig_position( [0.657 0.506  0.293 0.467]   );
    figure( data(3).hq  ); set_fig_position( [0.371 0.0426 0.293 0.467]   );
    figure( data(4).hq  ); set_fig_position( [0.657 0.0444 0.293 0.467]   );  
    
    figure( data(1).hqdot  ); set_fig_position( [0.367 0.506  0.293 0.468]   );
    figure( data(2).hqdot  ); set_fig_position( [0.657 0.506  0.293 0.467]   );
    figure( data(3).hqdot  ); set_fig_position( [0.371 0.0426 0.293 0.467]   );
    figure( data(4).hqdot );  set_fig_position( [0.657 0.0444 0.293 0.467]   );  
    
    % handle for figures
    for j=1:length(data)
        h(j).q      = data(j).hq;
        h(j).qshade = [];%Hq{j};
        
        h(j).qdot      = data(j).hqdot;
        h(j).qdotshade = [];%Hqdot{j};        
    end
    
    
    
end