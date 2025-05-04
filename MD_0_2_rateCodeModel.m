%% This script train a rate code-based classifier as a comparable model
% to the proposed double-layer multi-temporal resolution model
%
% Author: Xiwei She, PhD

clear; clc;
addpath(genpath('toolbox'));

%% Define cases
runCase = 1; % 1: SR; 2: MR; 3: Shifted; 4: Shuffled

categoryPool = 1:5; % Define Memory Label Here

% 10-25-50-1000 bin size corresponds to 20-50-100-2000 ms
% spikeBinSize = 10; binTime = 20;
% spikeBinSize = 25; binTime = 50;
% spikeBinSize = 50; binTime = 100;
spikeBinSize = 1000; binTime = 2000;

%% Modeling
lambda_pool = power(exp(1), 0:-0.03:-9); % Lambda pool for regularization
nestedFold = 1:5;

MCC_rateCode = zeros(length(categoryPool), 1);


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

    % Run through all nested cv outer folds
    yTest_allFold = [];
    yPred_allFold = [];
    for currentFold = nestedFold
        load(['processedData\', thisCase, '\MD_', Category, '_split', mat2str(currentFold)]);

        % Training

        % Bin the spikes based on small time window as the rate code
        reshaped_inputFeature_train = reshape(TrainingSet_SpikeTensor(:, 1:end-1, :), size(TrainingSet_SpikeTensor, 1), spikeBinSize, [], size(TrainingSet_SpikeTensor, 3));
        binned_average_train = mean(reshaped_inputFeature_train, 2);
        inputFeature_rateCode = reshape(binned_average_train, size(TrainingSet_SpikeTensor, 1), size(binned_average_train, 3) * size(TrainingSet_SpikeTensor, 3));

        outputTarget_rateCode = TrainingSet_target;
        [B, FitInfo] = lassoglm(inputFeature_rateCode, outputTarget_rateCode, 'binomial','CV', 5, 'Lambda', lambda_pool);
        idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
        mincoefs = B(:,idxLambdaMinDeviance);
        B0 = FitInfo.Intercept(idxLambdaMinDeviance);

        % Test

        % Bin the spikes based on small time window as the rate code
        reshaped_inputFeature_test = reshape(TestingSet_SpikeTensor(:, 1:end-1, :), size(TestingSet_SpikeTensor, 1), spikeBinSize, [], size(TestingSet_SpikeTensor, 3));
        binned_average_test = mean(reshaped_inputFeature_test, 2);
        testFeature_rateCode = reshape(binned_average_test, size(TestingSet_SpikeTensor, 1), size(binned_average_test, 3) * size(TestingSet_SpikeTensor, 3));

        testTarget_rateCode = TestingSet_target;
        y_i_testing = testFeature_rateCode * mincoefs + B0;
        yProb_testing = 1 ./ (1 + exp(-y_i_testing));

        yTest_allFold = [yTest_allFold; testTarget_rateCode];
        yPred_allFold = [yPred_allFold; double(yProb_testing>0.5)];
    end

    MCC_rateCode(ca, 1) = mcc(confusionmat(yTest_allFold, yPred_allFold));
end

disp(['MCC = ', mat2str(MCC_rateCode'), 'for A, B, P, T, V, respectively'])