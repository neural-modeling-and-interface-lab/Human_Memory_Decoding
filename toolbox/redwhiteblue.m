function C = redwhiteblue(vmin, vmax, m)
% This function change the colormap of the plot
% It visualize positive values as red color; negative values as blue color; 
% and zero as white color
% The VMIN and VMAX defines the ranges of the colors

if nargin == 2
    m = size(get(gcf, 'colormap'), 1);
elseif nargin ~= 3
    error('REDWHITEBLUE requires 2 or 3 parameters');
end
if vmin > vmax
    error('vmin should be less than vmax');
end
M = max(abs(vmin), abs(vmax));
% From [0 0 1] to [1 1 1] to [1 0 0];
color_range = linspace(vmin,vmax, m)';
R = interp1([-M 0 M],[0 1 1], color_range);
G = interp1([-M 0 M],[0 1 0], color_range);
B = interp1([-M 0 M],[1 1 0], color_range);
C = [R G B];