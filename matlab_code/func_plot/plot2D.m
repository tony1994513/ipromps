function [hp] = plot2D(hp, xmean, xcovii, ymean, ycovii, nSamples)


    style   = struct('Color', [0.6 0.6 0.6], 'LineStyle', '-', 'LineWidth', 2);
    style2  = struct('Color', [0.8 0.8 0.8], 'LineStyle', 'none', 'Marker', 'o', 'LineWidth', 2);
    
    x_sig = 1*sqrt(xcovii);
    y_sig = 1*sqrt(ycovii);
    
    nx = length(xmean);
   
    for k=1:nSamples
        try
            delete (hp)
        end
        xep = x_sig'.*randn(1, nx);
        yep = y_sig'.*randn(1, nx);
        
        
        x = xmean+xep;
        y = ymean+yep;
        
        facke_var = linspace(0,1,nx);
        desired   = linspace(0,1,400);
        
        xint = interp1(facke_var, x, desired);
        yint = interp1(facke_var, y, desired);
        
        
        
        hp(k) = plot( xint, yint, style2);
    end
    hp(k+1) = plot(xmean, ymean, style);
    
   
    
end