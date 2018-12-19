function [hxy] = plot2DwrapB(hxy, kf)
    
    
    for i=1:2
        
        xmean = kf(2*i-1).q_mean;
        ymean = kf(2*i).q_mean;

        xcovii = kf(2*i-1).q_Sigma_ii';
        ycovii = kf(2*i).q_Sigma_ii';

       
        figure(hxy(i).xy);
        dots_handle = plot2D(hxy(i).dots, xmean, xcovii, ymean, ycovii, 100);

        hxy(i).dots = dots_handle;
    end
    
    
end