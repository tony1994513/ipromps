function [hxy] = plot2DwrapC(hxy, kf, basis)
    
    
    for i=1:2
        
        xmean  = kf(2*i-1).w_mean;
        ymean  = kf(2*i).w_mean;

        xcov = kf(2*i-1).w_Sigma';
        ycov = kf(2*i).w_Sigma';
        
        
        %xCholSigma = chol(xcov);
        %yCholSigma = chol(ycov);

        
        xCholSigma = (xcov);
        yCholSigma = (ycov);

        
        
        figure(hxy(i).xy);
        dots_handle = plot2C(hxy(i).dots, xmean, xCholSigma, ymean, yCholSigma, basis, 100);

        hxy(i).dots = dots_handle;
    end
    
    
end