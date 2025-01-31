%% This script calculate the sparse classification functional matrix (SCFM)
% based on the modeling outcomes
%
% Author: Xiwei She
clear; clc;

categoryPool = 1:5; % 1:5;

nestedFold = 1:5;
num_split = 8;
lambda_pool = power(exp(1), 0:-0.03:-9);
resolution_pool = [0:25, 50:5:100];

runCase = 1; % 1: SR; 2: MR; 3: Shifted Control; 4: Shuffle Control

order = 3;
L = 1000;

switch (runCase)
    case 1
        thisCase = '1 Sample Response';
    case 2
        thisCase = '2 Match Response';
    case 3
        thisCase = '3 Shifted Control';
    case 4
        thisCase = '4 Shuffle Control';
    otherwise
        thisCase = 'UNDEFINED CASE!';
end

% Load neural data
iF01 = strcat('exampleData\ChannelInfo.mat');
load(iF01, 'CA3_Channels', 'CA1_Channels');
numNeuron = length(CA3_Channels) + length(CA1_Channels);

iF02 = strcat('exampleData\NeuralData.mat');
load(iF02, 'X', 'Y');

for ca = categoryPool
    switch ca
        case 1
            Category = 'Animal';
        case 2
            Category = 'Building';
        case 3
            Category = 'Plant';
        case 4
            Category = 'Tool';
        case 5
            Category = 'Vehicle';
    end

    % Load global optimal coeff for calculating SCFM 
    B_global_1 = cell(length(resolution_pool), 1);
    C0_global_1 = cell(length(resolution_pool), 1);
    for fold = nestedFold
        iF1 = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_fold', mat2str(fold), '_Parameters.mat');
        load(iF1, 'B_global', 'C0_global');

        for res = 1:length(resolution_pool)
            if fold == 1
                B_global_1{res} = B_global{res};
                C0_global_1{res} = C0_global{res};
            elseif fold == length(nestedFold)
                B_global_1{res} = B_global_1{res} + B_global{res};
                C0_global_1{res} = C0_global_1{res} + C0_global{res};

                B_global_1{res} = B_global_1{res} / length(nestedFold);
                C0_global_1{res} = C0_global_1{res} / length(nestedFold);
            else
                B_global_1{res} = B_global_1{res} + B_global{res};
                C0_global_1{res} = C0_global_1{res} + C0_global{res};
            end
        end
    end

    B_global_2_all = zeros(length(resolution_pool), length(nestedFold));
    C0_global_2_all = zeros(length(resolution_pool), length(nestedFold));
    for fold = nestedFold
        iF2 = strcat('result\MD_metalearner_',thisCase(3:end),'_', Category,'_fold', mat2str(fold),'_Parameters.mat');
        load(iF2, 'B_global', 'C0_global');

        B_global_2_all(:, fold) = B_global;
        C0_global_2_all(:, fold) = C0_global;
    end
    B_global_2 = mean(B_global_2_all, 2); C0_global_2 = mean(mean(C0_global_2_all, 2));

    % First layer SCFM
    SCFM_firstLayer = zeros(length(resolution_pool), L+1, numNeuron);
    C0_firstLayer = zeros(length(resolution_pool), 1);
    for resolution = resolution_pool
        resolutionIndex = find(resolution_pool == resolution);

        J = resolution+order+1; % BSpline knots for current resolution

        % B-Spline tools - Eq. 8&9
        BSpline = bspline(order+1, resolution+2, L+1);
        weights_firstLayer = B_global_1{resolutionIndex};
        % Reshape the weights
        weights2_firstLayer = reshape(weights_firstLayer, J, []);
        % Get the Functional Matrics - Eq. 15
        F_maskTemp = BSpline * weights2_firstLayer;
        w0Temp = C0_global_1{resolutionIndex};

        % Transform first layer SCFM into probability
        SCFM_firstLayer(resolutionIndex, :, :) = 1 ./ (1+exp(-w0Temp-F_maskTemp));
        C0_firstLayer(resolutionIndex) = 1 ./ (1+exp(-w0Temp));

    end

    % Second Layer SCFM
    SCFM_firstLayer_permute = SCFM_firstLayer(:, :); % Resolution * [decodingWindow * Neuron]
    SCFM_second_permute = B_global_2' * SCFM_firstLayer_permute;
    SCFM_second_0 =  B_global_2' * C0_firstLayer; % Baseline of first-layer
    SCFM_second = reshape(SCFM_second_permute, L+1, []); % DecodingWindow * Neuron

    SCFM_second_prob = 1 ./ (1 + exp(-C0_global_2-SCFM_second) ); % With C0
    SCFM_second_prob_baseline = 1 ./ (1 + exp(-C0_global_2-SCFM_second_0) ); % Consider baseline of the first-layer

    SCFM_Prob_final = SCFM_second_prob';
    SCFM_Prob_final_vis = SCFM_Prob_final - SCFM_second_prob_baseline;

    %  Visualization
    % SCFM
    figure('Position', [50, 50, 800, 800]);
    pcolorfull(SCFM_Prob_final_vis);
%     caxis([-1, 1])
    colormap(bluewhitered)
    %         xlabel('Time (-1 to 1s)');
    %         ylabel('Neurons');
    %         title(['SCFM (', Category, ')']);
    set(gca, 'box', 'on', 'fontSize', 24, 'fontName', 'Arial')

    % Save the SCFM
    oF = strcat('result\MD_SCFM_',thisCase(3:end), '_', Category,'.mat');
    save(oF, 'SCFM_Prob_final_vis')

end
