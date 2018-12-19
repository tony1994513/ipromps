function [style] = SBLACK(varargin)

    if isempty(varargin)
        lw = 1;
    else
        lw = varargin{1};
    end
    style = struct('Color', 'k', 'LineStyle', '-', 'LineWidth', lw);
    
end