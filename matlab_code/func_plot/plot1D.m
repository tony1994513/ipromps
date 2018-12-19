function [hh] = plot1D(h, x, fmean, fcovii, nSamples)

    style   = struct('Color', 'b', 'LineStyle', '.', 'LineWidth', 2);

    f_sig = 1*sqrt(fcovii);
   
    nx = length(fmean);

    for k=1:nSamples
        fep = f_sig.*randn(1, nx);
        hh(k) = plot(x , fmean+fep, style);
    end
    
end