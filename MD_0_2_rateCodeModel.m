%% This script train a rate code-based classifier as a comparable model
% to the proposed double-layer multi-temporal resolution model
%
% Author: Xiwei She, PhD

clear; clc;
addpath(genpath('toolbox'));

%% Define cases
runCase = 1; % 1: SR; 2: MR; 3: Shifted; 4: Shuffled

categoryPool = 1:5; % Define Memory Label Here

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
        inputFeature_rateCode = squeeze(mean(TrainingSet_SpikeTensor, 2)); % Averaged spike frequency / firing rate
        outputTarget_rateCode = TrainingSet_target;
        [B, FitInfo] = lassoglm(inputFeature_rateCode, outputTarget_rateCode, 'binomial','CV', 5, 'Lambda', lambda_pool);
        idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
        mincoefs = B(:,idxLambdaMinDeviance);
        B0 = FitInfo.Intercept(idxLambdaMinDeviance);

        % Test
        testFeature_rateCode = squeeze(mean(TestingSet_SpikeTensor, 2));
        testTarget_rateCode = TestingSet_target;
        y_i_testing = testFeature_rateCode * mincoefs + B0;
        yProb_testing = 1 ./ (1 + exp(-y_i_testing));

        yTest_allFold = [yTest_allFold; testTarget_rateCode];
        yPred_allFold = [yPred_allFold; double(yProb_testing>0.5)];
    end

    MCC_rateCode(ca, 1) = mcc(confusionmat(yTest_allFold, yPred_allFold));
end

disp(['MCC = ', mat2str(MCC_rateCode'), 'for A, B, P, T, V, respectively'])