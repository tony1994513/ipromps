function [h] = plotIterative(h, data, shadeColor)


    nJoints = length(data);
    nTraj   = length(data(1).q);
    
    
    for j=1:nJoints
        
        figure(h(j).q);
        
        try
            delete(h(j).shade.mainLine);
        end
        try
            delete(h(j).shade.patch);
        end
        try
            delete(h(j).shade.edge);
        end
        
        h(j).shade = shadedErrorBar([1:nTraj], data(j).q,...
                                    2*sqrt( data(j).P_ii  ), ...
                                  {shadeColor, 'LineWidth', 2'}, 1);   
                              

    end
    
end