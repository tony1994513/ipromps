function [hp] = plot2C(hp, xmean, xcovii, ymean, ycovii, basis, nSamples)


    style   = struct('Color', 'b', 'LineStyle', '-', 'LineWidth', 2);
    style2  = struct('Color', 'g', 'LineStyle', '-', 'LineWidth', 1);
    
    if 1
        x_sig = sqrt(xcovii);
        y_sig = sqrt(ycovii);
    else
        x_sig = (xcovii);
        y_sig = (ycovii);
    end
    nx = length(xmean);
   
    for k=1:nSamples
    
        % sampling a noise value from the normal distribution (randn)
        
        xep =  x_sig*randn(nx, 1);
        yep =  y_sig*randn(nx, 1);
        
        
%         xep = diag( x_sig*randn(nx,nx) ); 
%         yep = diag( y_sig*randn(nx,nx) );
        
        
        % a new vector of weights for the x and y
        x.q  = basis.Gn*(xmean+xep);
        y.q  = basis.Gn*(ymean+yep);
        
        x.qmean  = basis.Gn*(xmean);
        y.qmean  = basis.Gn*(ymean);
        
        
        x.qnoise = basis.Gn*diag(xep.^2)*basis.Gn';
        y.qnoise = basis.Gn*diag(yep.^2)*basis.Gn';
        
        x.qnoise = diag(x.qnoise);
        y.qnoise = diag(y.qnoise);
        
%         x.qdot  = basis.Gndot*x.w;
%         y.qdot  = basis.Gndot*y.w;
        

        try
            delete (hp)
        end        
        %hp(k) = plot( x.qmean + x.qnoise, y.qmean +y.qnoise, style2);
        hp(k) = plot( x.q , y.q, style2);
    end
    
    hp(k+1) = plot(x.qmean, y.qmean, style);
    
   
    
end