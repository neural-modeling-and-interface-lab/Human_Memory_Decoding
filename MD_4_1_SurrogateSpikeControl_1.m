%% This script estimate the model performance when input spikes were shuffled
% this is for answering one of the reviewer's question
clear; clc;

runCase = 1; % 1: SR; 2: MR; 3: Shifted; 4: Shuffled

categoryPool = 1:5; % Define Memory Label Here

nestedFold = 1:5;

resolution_all = [0:25, 50:5:100];
num_split = 8;
lambda_pool = power(exp(1), 0:-0.03:-9); % Define the lambda pool

runTime = 100; % Shuffling times

switch (runCase)
    case 1
        thisCase = '1 Sample Response';
    case 2
        thisCase = '2 Match Response';
    case 3
        thisCase = '3 SR&MR';
    case 4
        thisCase = '4 Shifted Control';
    case 5
        thisCase = '5 Shuffle Control';
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



    %% First layer predicting (using surrogate spikes + trained model)

    yProb_training = cell(length(MD_baselearner.resolution_all), length(nestedFold), runTime);
    yTrue_training = cell(length(MD_baselearner.resolution_all), length(nestedFold), runTime);
    yProb_testing = cell(length(MD_baselearner.resolution_all), length(nestedFold), runTime);
    yTrue_testing = cell(length(MD_baselearner.resolution_all), length(nestedFold), runTime);
    P_testing_fold = cell(length(MD_baselearner.resolution_all), length(nestedFold), runTime);
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

            % Start testing using surrogate spikes generated randomly
            % across run times
            for runTemp = 1:runTime

                % Preparing for generating the control inputs
                spikeTensor_train_0 = MD_baselearner.TrainingSet_SpikeTensor;
                spikeTensor_test_0 = MD_baselearner.TestingSet_SpikeTensor;

                % Here, for each neuron and trial we shuffle the input
                % spike trains by changing the starting time and
                % rolling over the remaining spike to the initial portion
                spikeTensor_train_control_1 = zeros(size(spikeTensor_train_0));
                for trialIndex = 1:size(spikeTensor_train_0, 1)
                    for neuronIndex = 1:size(spikeTensor_train_0, 3)
                        shuffleStep_control_1 = randi(size(spikeTensor_train_0, 2));

                        tempContent = spikeTensor_train_0(trialIndex, :, neuronIndex);
                        contentForRolling_part_1 = tempContent(1:shuffleStep_control_1);
                        contentForRolling_part_2 = tempContent(shuffleStep_control_1+1 : end);
                        spikeTensor_train_control_1(trialIndex, :, neuronIndex) = [contentForRolling_part_2, contentForRolling_part_1];
                    end
                end

                spikeTensor_test_control_1 = zeros(size(spikeTensor_test_0));
                for trialIndex = 1:size(spikeTensor_test_0, 1)
                    for neuronIndex = 1:size(spikeTensor_test_0, 3)
                        shuffleStep_control_1 = randi(size(spikeTensor_test_0, 2));

                        tempContent = spikeTensor_test_0(trialIndex, :, neuronIndex);
                        contentForRolling_part_1 = tempContent(1:shuffleStep_control_1);
                        contentForRolling_part_2 = tempContent(shuffleStep_control_1+1 : end);
                        spikeTensor_test_control_1(trialIndex, :, neuronIndex) = [contentForRolling_part_2, contentForRolling_part_1];
                    end
                end


                % Nested outer trainning
                P_training = SpikeTensor2BSplineFeatureMatrix(spikeTensor_train_control_1, MD_baselearner.resolution_all(resolution), MD_baselearner.d);
                y_i_training = P_training * B_global{resolution, 1} + C0_global{resolution, 1};
                yProb_training{resolution, currentFold, runTemp} = 1 ./ (1 + exp(-y_i_training));
                yTrue_training{resolution, currentFold, runTemp} = MD_baselearner.TrainingSet_target;

                % Nested outer testing
                P_testing = SpikeTensor2BSplineFeatureMatrix(spikeTensor_test_control_1, MD_baselearner.resolution_all(resolution), MD_baselearner.d);
                y_i_testing = P_testing * B_global{resolution, 1} + C0_global{resolution, 1};
                yProb_testing{resolution, currentFold, runTemp} = 1 ./ (1 + exp(-y_i_testing));
                yTrue_testing{resolution, currentFold, runTemp} = MD_baselearner.TestingSet_target;
                P_testing_fold{resolution, currentFold, runTemp} = P_testing;

            end

        end
    end

    oF = strcat('result\MD_baselearner_',thisCase(3:end), '_',Category,'_fold',mat2str(currentFold), '_control_1_outputs.mat');
    save(oF, 'yProb_training', 'yProb_testing');


    %% Meta learner (using base learners outputs from surrogate spikes)
    % numFold x numSplit x numLambda
    deviance_allFolds = zeros(length(nestedFold), num_split, length(lambda_pool));
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
    global_lambda = lambda_pool(minIndex);

    % Load base learner results
    iF_2 = strcat('result\MD_baselearner_',thisCase(3:end), '_',Category,'_fold',mat2str(currentFold), '_control_1_outputs.mat');
    firstResults = load(iF_2, 'yProb_testing', 'yProb_training');
    bestSecondLayerMCC_testing = zeros(runTime, 1);

    for runTemp = 1:runTime

        % Model Evaluation - Second layer predicting
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
            P_training_0 = firstResults.yProb_training(:, currentFold, runTemp);

            P_training = zeros(size(P_training_0{1}, 1), size(B_global, 1));
            for tempI = 1:length(P_training_0)
                P_training(:, tempI) = P_training_0{tempI};
            end
            y_i_training = P_training * B_global + C0_global;
            yProb_training{currentFold} = 1 ./ (1 + exp(-y_i_training));
            yTrue_training{currentFold} = MD_metalearner.TrainingSet_target;

            % Nested outer testing
            P_testing_0 = firstResults.yProb_testing(:, currentFold, runTemp);

            P_testing = zeros(size(P_testing_0{1}, 1), size(B_global, 1));
            for tempI = 1:length(P_testing_0)
                P_testing(:, tempI) = P_testing_0{tempI};
            end
            y_i_testing = P_testing * B_global + C0_global;
            yProb_testing{currentFold} = 1 ./ (1 + exp(-y_i_testing));
            yTrue_testing{currentFold} = MD_metalearner.TestingSet_target;
            P_testing_fold{currentFold} = P_testing;

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


        % Visualization
        bestSecondLayerMCC_testing(runTemp) = MCC_testing;


    end
    disp(['Second layer overall outer testing MCC: ', mat2str(mean(bestSecondLayerMCC_testing)), ' +/- ', mat2str(std(bestSecondLayerMCC_testing))])
    disp('============================================================================')
    disp(' ')

end
