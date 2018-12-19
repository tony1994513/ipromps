function [style] = SBLUE(varargin)
    if isempty(varargin)
        lw = 1;
    else
        lw = varargin{1};
    end
    style = struct('Color', 'b', 'LineStyle', '-', 'LineWidth', lw);
end