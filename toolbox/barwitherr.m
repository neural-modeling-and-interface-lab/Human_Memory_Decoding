%**************************************************************************
%
%   This is a simple extension of the bar plot to include error bars.  It
%   is called in exactly the same way as bar but with an extra input
%   parameter "errors" passed first.
%
%   Parameters:
%   errors - the errors to be plotted (extra dimension used if assymetric)
%   varargin - parameters as passed to conventional bar plot
%   See bar and errorbar documentation for more details.
%
%   Output:
%   [hBar hErrorbar] = barwitherr(..) returns a vector of handles to the
%                      barseries (hBar) and error bar (hErrorbar) objects
%
%   Symmetric Example:
%   y = randn(3,4);         % random y values (3 groups of 4 parameters)
%   errY = 0.1.*y;          % 10% error
%   h = barwitherr(errY, y);% Plot with errorbars
%
%   set(gca,'XTickLabel',{'Group A','Group B','Group C'})
%   legend('Parameter 1','Parameter 2','Parameter 3','Parameter 4')
%   ylabel('Y Value')
%   set(h(1),'FaceColor','k');
%
%
%   Asymmetric Example:
%   y = randn(3,4);         % random y values (3 groups of 4 parameters)
%   errY = zeros(3,4,2);
%   errY(:,:,1) = 0.1.*y;   % 10% lower error
%   errY(:,:,2) = 0.2.*y;   % 20% upper error
%   barwitherr(errY, y);    % Plot with errorbars
%
%   set(gca,'XTickLabel',{'Group A','Group B','Group C'})
%   legend('Parameter 1','Parameter 2','Parameter 3','Parameter 4')
%   ylabel('Y Value')
%
%
%   Notes:
%   Ideally used for group plots with non-overlapping bars because it
%   will always plot in bar centre (so can look odd for over-lapping bars)
%   and for stacked plots the errorbars will be at the original y value is
%   not the stacked value so again odd appearance as is.
%
%   The data may not be in ascending order.  Only an issue if x-values are
%   passed to the fn in which case their order must be determined to
%   correctly position the errorbars.
%
%
%   24/02/2011  Martina F. Callaghan    Created
%   12/08/2011  Martina F. Callaghan    Updated for random x-values
%   24/10/2011  Martina F. Callaghan    Updated for asymmetric errors
%   15/11/2011  Martina F. Callaghan    Fixed bug for assymetric errors &
%                                       vector plots
%   14/06/2013  Martina F. Callaghan    Returning handle as recommended by
%                                       Eric (see submission comments)
%   08/07/2013  Martina F. Callaghan    Only return handle if requested.
%   18/07/2013  Martina F. Callaghan    Bug fix for single group data that
%                                       allows assymetric errors.
%                                       Also removed dot from display as
%                                       per Charles Colin comment. The
%                                       handle can be returned to control
%                                       appearance.
%   27/08/2013  Martina F. Callaghan    Ensuring errors are always stored
%                                       as lowerErrors and upperErrors even
%                                       if symmetric.
%   29/10/2014  Martina F. Callaghan    Updated for 2014b graphics
%
%**************************************************************************

function varargout = barwitherr(errors,varargin)

% Check how the function has been called based on requirements for "bar"
if nargin < 4
    % This is the same as calling bar(y)
    values = varargin{1};
    xOrder = 1:size(values,1);
    %     distributionInfo = round(varargin{2});
    %     varargin(2) = [];
else
    % This means extra parameters have been specified
    if isscalar(varargin{2}) || ischar(varargin{2})
        % It is a width / property so the y values are still varargin{1}
        values = varargin{1};
        xOrder = 1:size(values,1);
    else
        % x-values have been specified so the y values are varargin{2}
        % If x-values have been specified, they could be in a random order,
        % get their indices in ascending order for use with the bar
        % locations which will be in ascending order:
        values = varargin{2};
        [tmp xOrder] = sort(varargin{1});
    end
end

% If an extra dimension is supplied for the errors then they are
% assymetric split out into upper and lower:
if ndims(errors) == ndims(values)+1
    lowerErrors = errors(:,:,1);
    upperErrors = errors(:,:,2);
elseif isvector(values)~=isvector(errors)
    lowerErrors = errors(:,1);
    upperErrors = errors(:,2);
else
    lowerErrors = errors;
    upperErrors = errors;
end


% Check that the size of "errors" corresponsds to the size of the y-values.
% Arbitrarily using lower errors as indicative.
if any(size(values) ~= size(lowerErrors))
    error('The values and errors have to be the same length')
end

[nRows nCols] = size(values);
handles.bar = bar(varargin{:}); % standard implementation of bar fn

tempColor = linspecer(length(handles.bar), 'sequential');
% tempColor = jet(length(handles.bar));
for i = 1:length(handles.bar)
    set(handles.bar(i),'FaceColor', tempColor(i, :))
end
set(handles.bar, 'EdgeColor', 'k')

hold on
hBar = handles.bar;

if nRows > 1
    hErrorbar = zeros(1,nCols);
    for col = 1:nCols
        % Extract the x location data needed for the errorbar plots:
        if verLessThan('matlab', '8.4')
            % Original graphics:
            x = get(get(handles.bar(col),'children'),'xdata');
        else
            % New graphics:
            x =  handles.bar(col).XData + [handles.bar(col).XOffset];
        end
          
        % Added by Xiwei - For those didn't pass the t-test, ploting a Red circle marker instead
        temp3 = (lowerErrors(xOrder, col) == -2);
        temp4 = mean(x,1);
        len = length(temp4(temp3));
        errorbar(temp4(temp3), zeros(len,1), zeros(len,1), zeros(len,1), 'or', 'LineWidth', 1.5);
        lowerErrors(temp3,col) = 0;upperErrors(temp3, col) = 0;        
        
        % Added by Xiwei - For those doesn't have enough trials, ploting a Black circle marker instead
        temp1 = (values(xOrder, col) == 0);
        temp1(temp3) = 0;
        temp2 = mean(x,1);
        len = length(temp2(temp1));
        errorbar(temp2(temp1), zeros(len,1), zeros(len,1), zeros(len,1), 'ok', 'LineWidth', 1.5);
        lowerErrors(temp1,col) = 0;upperErrors(temp1, col) = 0;
        
        % Added by Xiwei - For those negative, ploting a Red circle marker instead
%         temp3 = errors(xOrder, col) < 0;
%         temp4 = mean(x,1);
%         len = length(temp4(temp3));
%         errorbar(temp4(temp3), zeros(len,1), zeros(len,1), zeros(len,1), 'or', 'LineWidth', 1.5);

        % Use the mean x values to call the standard errorbar fn; the
        % errorbars will now be centred on each bar; these are in ascending
        % order so use xOrder to ensure y values and errors are too:
        hErrorbar(col) = errorbar(mean(x,1), values(xOrder,col), lowerErrors(xOrder,col), upperErrors(xOrder, col), '.k');
        set(hErrorbar(col), 'marker', 'none', 'CapSize', 6, 'LineWidth', 1.5)
%         set(hErrorbar(col), 'Color', tempColor(col, :))
    end
else
    if verLessThan('matlab', '8.4')
        % Original graphics:
        x = get(get(handles.bar,'children'),'xdata');
    else
        % New graphics:
        x =  handles.bar.XData + [handles.bar.XOffset];
    end
    
    hErrorbar = errorbar(mean(x,1), values, lowerErrors, upperErrors, '.k');
    set(hErrorbar, 'marker', 'none', 'CapSize', 6, 'LineWidth', 1.5)
end

hold off

switch nargout
    case 1
        varargout{1} = hBar;
    case 2
        varargout{1} = hBar;
        varargout{2} = hErrorbar;
end