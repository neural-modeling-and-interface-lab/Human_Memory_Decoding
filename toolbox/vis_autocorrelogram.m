function vis_autocorrelogram(SpikeTensor)

for j = 1:size(SpikeTensor, 3)
    % Variable "SpikeTensor" is loaded from data file
    % Binary values, dimensionality as: Trial * Time * Chan
    temp = SpikeTensor(:, :, j); % Pick each channel
    temp2 = temp(:);
    spike_times = find(temp2~=0)' / 500; % spike times (in seconds)

    % Parameters
    max_lag = 0.1;  % Maximum lag in seconds
    bin_size = 0.001;  % Bin size in seconds

    % Create time bins
    edges = -max_lag:bin_size:max_lag;

    % Calculate autocorrelogram
    n_bins = length(edges) - 1;  % Number of bins
    autocorr = zeros(1, n_bins);

    % Loop through each spike and count spikes in the bins
    for i = 1:length(spike_times)
        % Calculate the difference from the current spike
        diffs = spike_times - spike_times(i);

        % Count spikes in each bin
        counts = histcounts(diffs, edges);

        % Accumulate the counts
        autocorr = autocorr + counts;
    end

    % Normalize by the number of spikes and the bin size
    autocorr = autocorr / (length(spike_times) * bin_size);

    % Exclude the zero-lag part from the plotting
    zero_lag_index = find(edges >= 0, 1); % Find the index of zero lag
    edges_no_zero = edges([1:zero_lag_index-1, zero_lag_index+1:end]); % Exclude zero lag edge
    autocorr_no_zero = autocorr([1:zero_lag_index-1, zero_lag_index+1:end]); % Exclude zero lag counts

    % Plotting
    figure;
    bar(edges_no_zero(1:end-1), autocorr_no_zero, 'histc'); % 'histc' aligns the bars with edges
    xlabel('Time Lag (s)');
    ylabel('Autocorrelogram');
    xlim([-max_lag, max_lag]);

end