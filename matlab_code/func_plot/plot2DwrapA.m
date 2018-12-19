function [h] = plot2DwrapA(h, data)


    xmean = data(1).q_mean;
    ymean = data(2).q_mean;
    
    xcovii = diag(data(1).q_cov);
    ycovii = diag(data(2).q_cov);
    
    h = plot2D(h, xmean, xcovii, ymean, ycovii, 100);
  
    nD = size(data(1).q,1);
    for k=1:nD
        plot(data(1).q(k,:), data(2).q(k,:), 'm', 'LineWidth', 2);
    end
    
end