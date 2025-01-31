%% This script calculate the global optimal parameters
% and measures the performance of the SECOND layer (meta learner)
%
% Author: Xiwei She, PhD
clear; clc;
addpath(genpath('toolbox'));
runCase = 1; % 1: SR; 2: MR; 3: Shifted Control; 4: Shuffle Control

categoryPool = 1:5;

nestedFold = 1:5;
num_split = 8;
lambda_pool_2 = power(exp(1), 0:-0.03:-9); % Define the lambda pool

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

MCC_1_overall = zeros(1, length(categoryPool));
MCC_2_overall = zeros(1, length(categoryPool));

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

    printStr = ['Processing case ', mat2str(runCase), ' & ', Category];
    disp(printStr);

    %% Model Selection - Find global optimal model parameters
    % numFold x numSplit x numLambda
    deviance_allFolds = zeros(length(nestedFold), num_split, length(lambda_pool_2));
    B_allFolds = cell(length(nestedFold), num_split);
    FitInfo_allFolds = cell(length(nestedFold), num_split);
    for currentFold = nestedFold

        iF = strcat('result\MD_metalearner_',thisCase(3:end), '_', Category,'_fold',mat2str(currentFold), '.mat');
        load(iF);

        % Loop through all bagging splits
        for split = 1:num_split
            thisR = MD_metalearner.R_second(split);
            deviance_allFolds(currentFold, split, :) = thisR{1}.FitInfo_second.Deviance;
            B_allFolds{currentFold, split} = thisR{1}.B_second;
            FitInfo_allFolds{currentFold, split} = thisR{1}.FitInfo_second;
        end

    end

    % Find the global optimal lambda index cross all folds and split
    deviance_ave = squeeze(sum(sum(deviance_allFolds, 1), 2));
    minIndex = find(deviance_ave == min(deviance_ave));
    if length(minIndex) > 1
        minIndex = minIndex(1);
    end
    global_minDevianceIndices = minIndex;
    global_lambda = lambda_pool_2(minIndex);

    %% Model Evaluation - Meta learner predicting
    yProb_training = cell(length(nestedFold), 1);
    yTrue_training = cell(length(nestedFold), 1);
    yProb_testing = cell(length(nestedFold), 1);
    yTrue_testing = cell(length(nestedFold), 1);
    P_testing_fold = cell(length(nestedFold), 1);
    for currentFold = nestedFold

        iF = strcat('result\MD_metalearner_',thisCase(3:end), '_', Category,'_fold',mat2str(currentFold), '.mat');
        load(iF);

        % Calculate the averaged coefficient by using the global lambda
        thisB = cell(num_split, 1);
        thisC0 = cell(num_split, 1);
        for split = 1:num_split
            thisB{split} = B_allFolds{currentFold, split}(:, global_minDevianceIndices);
            thisC0{split} = FitInfo_allFolds{currentFold, split}.Intercept(global_minDevianceIndices);
        end
        B_global_temp = zeros(size(thisB{1}));
        C0_global_temp = zeros(size(thisC0{1}));

        for split = 1:num_split
            B_global_temp = B_global_temp + thisB{split};
            C0_global_temp = C0_global_temp + thisC0{split};
        end

        B_global = B_global_temp / num_split;
        C0_global = C0_global_temp / num_split;

        % Nested outer trainning
        P_training = MD_metalearner.yProb_training;
        y_i_training = P_training * B_global + C0_global;
        yProb_training{currentFold} = 1 ./ (1 + exp(-y_i_training));
        yTrue_training{currentFold} = MD_metalearner.TrainingSet_target;

        % Nested outer testing
        P_testing = MD_metalearner.yProb_testing;
        y_i_testing = P_testing * B_global + C0_global;
        yProb_testing{currentFold} = 1 ./ (1 + exp(-y_i_testing));
        yTrue_testing{currentFold} = MD_metalearner.TestingSet_target;
        P_testing_fold{currentFold} = P_testing;

        % Save for SCFM calculations
        oF = strcat('result\MD_metalearner_',thisCase(3:end), '_', Category,'_fold',mat2str(currentFold), '_Parameters.mat');
        save(oF, 'B_global', 'C0_global');
    end

    % Overall performance - long vertor of all folds
    yProb_training_long = [];
    yTrue_training_long = [];
    yProb_testing_long = [];
    yTrue_testing_long = [];
    P_testing_long = [];
    for currentFold = nestedFold
        yProb_training_long = [yProb_training_long;  yProb_training{currentFold}];
        yTrue_training_long = [yTrue_training_long;  yTrue_training{currentFold}];
        yProb_testing_long = [yProb_testing_long;  yProb_testing{currentFold}];
        yTrue_testing_long = [yTrue_testing_long;  yTrue_testing{currentFold}];
        P_testing_long = [P_testing_long;  P_testing_fold{currentFold}];
    end

    CM_training_temp = confusionmat(double(yProb_training_long>0.5), yTrue_training_long);
    if (size(CM_training_temp,1)==1&&size(CM_training_temp,2)==1)
        CM_training_temp = [CM_training_temp(1,1) 0;0 0];
    end
    CM_training = CM_training_temp;
    MCC_training = mcc(CM_training_temp);

    CM_testing_temp = confusionmat(double(yProb_testing_long>0.5), yTrue_testing_long);
    if (size(CM_testing_temp,1)==1&&size(CM_testing_temp,2)==1)
        CM_testing_temp = [CM_testing_temp(1,1) 0;0 0];
    end
    CM_testing = CM_testing_temp;
    MCC_testing = mcc(CM_testing_temp);

    TN = CM_testing(1, 1); TP = CM_testing(2, 2);
    FN = CM_testing(1, 2); FP = CM_testing(2, 1);
    Sensitivity = TP / (TP+FN);
    Specificity = TN / (TN+FP);

    % Load First Layer Results
    iF2 = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_Performance.mat');
    load(iF2, 'bestFirstLayerMCC_training', 'bestFirstLayerMCC_testing');

    % Visualization
    bestSecondLayerMCC_training = MCC_training;
    bestSecondLayerMCC_testing = MCC_testing;

    MCC_1_overall(ca) = bestFirstLayerMCC_testing;
    MCC_2_overall(ca) = bestSecondLayerMCC_testing;

    disp('======================== DLMDM Results Summary ========================')
    disp(['First layer overall outer training MCC: ', mat2str(bestFirstLayerMCC_training)])
    disp(['First layer overall outer testing MCC: ', mat2str(bestFirstLayerMCC_testing)])
    disp(['Second layer overall outer training MCC: ', mat2str(bestSecondLayerMCC_training)])
    disp(['Second layer overall outer testing MCC: ', mat2str(bestSecondLayerMCC_testing)])
    disp(['Second layer overall outer testing Sensitivity: ', mat2str(Sensitivity)])
    disp(['Second layer overall outer testing Specificity: ', mat2str(Specificity)])
    disp('============================================================================')
    disp(' ')

    % Save Results
    oF2 = strcat('result\MD_metalearner_',thisCase(3:end), '_', Category,'_Performance.mat');
    save(oF2, 'bestFirstLayerMCC_training', 'bestFirstLayerMCC_testing', 'bestSecondLayerMCC_training', 'bestSecondLayerMCC_testing', 'yProb_testing_long', 'yTrue_testing_long', 'P_testing_long');

end