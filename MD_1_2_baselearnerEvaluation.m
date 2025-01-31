%% This script measures the performance of first layer base learners
% and calculate the global optimal parameters for the second layer (meta-larner)
%
% Author: Xiwei She
clear; clc;
addpath(genpath('toolbox'));
categoryPool = 1:5; % 1:5;

runCase = 1; % 1: SR; 2: MR; 3: Shifted Control; 4: Shuffle Control

nestedFold = 1:5;

resolution_all = [0:25, 50:5:100];
num_split = 8;
lambda_pool = power(exp(1), 0:-0.03:-9);

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

    printStr = ['Processing case ', mat2str(runCase), ' & ', Category];
    disp(printStr);

    %% Model Selection - Find global optimal model parameters
    % numFold x numSplit x numResolution x numLambda
    deviance_allFolds = zeros(length(nestedFold), num_split, length(resolution_all), length(lambda_pool));
    B_allFolds = cell(length(nestedFold), num_split, length(resolution_all));
    FitInfo_allFolds = cell(length(nestedFold), num_split, length(resolution_all));
    for currentFold = nestedFold

        iF = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_fold',mat2str(currentFold), '.mat');
        load(iF);

        % Loop through all base learners (resolutions)
        for resolution = 1:length(MD_baselearner.resolution_all)
            % Loop through all bagging splits
            for split = 1:num_split
                thisR = MD_baselearner.R_first(split, resolution);
                deviance_allFolds(currentFold, split, resolution, :) = thisR{1}.FitInfo.Deviance;
                B_allFolds{currentFold, split, resolution} = thisR{1}.B;
                FitInfo_allFolds{currentFold, split, resolution} = thisR{1}.FitInfo;
            end
        end

    end

    % Find the global optimal lambda index cross all folds and split
    deviance_ave = squeeze(sum(sum(deviance_allFolds, 1), 2)); % sumed deviances of all base learners
    global_minDevianceIndices = zeros(length(MD_baselearner.resolution_all), 1);
    global_lambda = zeros(length(MD_baselearner.resolution_all), 1);
    for resolution = 1:length(MD_baselearner.resolution_all) % find min deviance of each base learner
        minIndex = find(deviance_ave(resolution, :) == min(deviance_ave(resolution, :)));
        if length(minIndex) > 1
            minIndex = minIndex(1);
        end

        % Check whether this global lambda shrink all coefficients to zeros
        % If so, use the nearest lambda with no all zero coef
        Coef = B_allFolds(:, :, resolution);
        allZeroCoef = 1;
        while(allZeroCoef == 1 && minIndex ~= 1)
            for currentFold = nestedFold
                for split = 1:num_split
                    tempCoef = Coef{currentFold, split}(:, minIndex);
                    if sum(tempCoef) ~= 0
                        allZeroCoef = 0;
                    end
                end
            end
            if allZeroCoef == 1
                minIndex = minIndex-1;
            end
        end

        global_minDevianceIndices(resolution, 1) = minIndex;
        global_lambda(resolution, 1) = lambda_pool(minIndex);
    end

    %% Model Evaluation - Base learner predicting
    yProb_training = cell(length(MD_baselearner.resolution_all), length(nestedFold));
    yTrue_training = cell(length(MD_baselearner.resolution_all), length(nestedFold));
    yProb_testing = cell(length(MD_baselearner.resolution_all), length(nestedFold));
    yTrue_testing = cell(length(MD_baselearner.resolution_all), length(nestedFold));
    P_testing_fold = cell(length(MD_baselearner.resolution_all), length(nestedFold));
    for currentFold = nestedFold

        iF = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_fold',mat2str(currentFold), '.mat');
        load(iF);

        B_global = cell(length(MD_baselearner.resolution_all), 1);
        C0_global = cell(length(MD_baselearner.resolution_all), 1);

        % Loop through all base learners (resolutions)
        for resolution = 1:length(MD_baselearner.resolution_all)

            % Calculate the averaged coefficient by using the global lambda
            thisB = cell(num_split, 1);
            thisC0 = cell(num_split, 1);
            for split = 1:num_split
                thisB{split} = B_allFolds{currentFold, split, resolution}(:, global_minDevianceIndices(resolution));
                thisC0{split} = FitInfo_allFolds{currentFold, split, resolution}.Intercept(global_minDevianceIndices(resolution));
            end
            B_global_temp = zeros(size(thisB{1}));
            C0_global_temp = zeros(size(thisC0{1}));

            for split = 1:num_split
                B_global_temp = B_global_temp + thisB{split};
                C0_global_temp = C0_global_temp + thisC0{split};
            end

            B_global{resolution, 1} = B_global_temp / num_split;
            C0_global{resolution, 1} = C0_global_temp / num_split;

            % Nested outer trainning
            P_training = SpikeTensor2BSplineFeatureMatrix(MD_baselearner.TrainingSet_SpikeTensor, MD_baselearner.resolution_all(resolution), MD_baselearner.d);
            y_i_training = P_training * B_global{resolution, 1} + C0_global{resolution, 1};
            yProb_training{resolution, currentFold} = 1 ./ (1 + exp(-y_i_training));
            yTrue_training{resolution, currentFold} = MD_baselearner.TrainingSet_target;

            % Nested outer testing
            P_testing = SpikeTensor2BSplineFeatureMatrix(MD_baselearner.TestingSet_SpikeTensor, MD_baselearner.resolution_all(resolution), MD_baselearner.d);
            y_i_testing = P_testing * B_global{resolution, 1} + C0_global{resolution, 1};
            yProb_testing{resolution, currentFold} = 1 ./ (1 + exp(-y_i_testing));
            yTrue_testing{resolution, currentFold} = MD_baselearner.TestingSet_target;
            P_testing_fold{resolution, currentFold} = P_testing;
        end

        oF = strcat('result\MD_baselearner_',thisCase(3:end), '_',Category,'_fold',mat2str(currentFold), '_Parameters.mat');
        save(oF, 'yProb_training', 'yProb_testing', 'B_global', 'C0_global');
    end

    % Overall performance - long vertor of all folds
    yProb_training_long = cell(length(MD_baselearner.resolution_all), 1);
    yTrue_training_long = cell(length(MD_baselearner.resolution_all), 1);
    yProb_testing_long = cell(length(MD_baselearner.resolution_all), 1);
    yTrue_testing_long = cell(length(MD_baselearner.resolution_all), 1);
    CM_training = cell(length(MD_baselearner.resolution_all), 1);
    CM_testing = cell(length(MD_baselearner.resolution_all), 1);
    MCC_training = zeros(length(MD_baselearner.resolution_all), 1);
    MCC_testing = zeros(length(MD_baselearner.resolution_all), 1);
    P_testing_long = cell(length(MD_baselearner.resolution_all), 1);
    for resolution = 1:length(MD_baselearner.resolution_all)
        for currentFold = nestedFold
            yProb_training_long{resolution} = [yProb_training_long{resolution};  yProb_training{resolution, currentFold}];
            yTrue_training_long{resolution} = [yTrue_training_long{resolution};  yTrue_training{resolution, currentFold}];
            yProb_testing_long{resolution} = [yProb_testing_long{resolution};  yProb_testing{resolution, currentFold}];
            yTrue_testing_long{resolution} = [yTrue_testing_long{resolution};  yTrue_testing{resolution, currentFold}];

            P_testing_long{resolution} = [P_testing_long{resolution}; P_testing_fold{resolution, currentFold}];
        end

        CM_training_temp = confusionmat(double(yProb_training_long{resolution}>0.5), yTrue_training_long{resolution});
        if (size(CM_training_temp,1)==1&&size(CM_training_temp,2)==1)
            CM_training_temp = [CM_training_temp(1,1) 0;0 0];
        end
        CM_training{resolution} = CM_training_temp;
        MCC_training(resolution) = mcc(CM_training_temp);

        CM_testing_temp = confusionmat(double(yProb_testing_long{resolution}>0.5), yTrue_testing_long{resolution});
        if (size(CM_testing_temp,1)==1&&size(CM_testing_temp,2)==1)
            CM_testing_temp = [CM_testing_temp(1,1) 0;0 0];
        end
        CM_testing{resolution} = CM_testing_temp;
        MCC_testing(resolution) = mcc(CM_testing_temp);
    end

    % Visualization
    [bestFirstLayerMCC_training, bestFirstLayerMCC_index] = max(MCC_training);
    bestFirstLayerMCC_training = bestFirstLayerMCC_training(1);
    bestFirstLayerMCC_index = bestFirstLayerMCC_index(1);
    bestFirstLayerMCC_testing = MCC_testing(bestFirstLayerMCC_index);

    CM_testing_0 = CM_testing{bestFirstLayerMCC_index};
    TN = CM_testing_0(1, 1); TP = CM_testing_0(2, 2);
    FN = CM_testing_0(1, 2); FP = CM_testing_0(2, 1);
    Sensitivity = TP / (TP+FN);
    Specificity = TN / (TN+FP);

    disp('======================== First Layer Results Summary ========================')
    disp(['Overall outer training MCC: ', mat2str(bestFirstLayerMCC_training)])
    disp(['Overall outer testing MCC: ', mat2str(bestFirstLayerMCC_testing)])
    disp(['Overall outer testing Sensitivity: ', mat2str(Sensitivity)])
    disp(['Overall outer testing Specificity: ', mat2str(Specificity)])
    disp('============================================================================')
    disp(' ')

    % Save Results
    oF2 = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_Performance.mat');
    save(oF2, 'bestFirstLayerMCC_training', 'bestFirstLayerMCC_testing', 'yProb_testing_long', 'yTrue_testing_long', 'P_testing_long');
end