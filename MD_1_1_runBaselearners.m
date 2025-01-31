%% This is a running script for training (the first layer) base learners 
% in the double-layer model using the class "MD_baselearner"
%
% It utilizes parallel computing strategy to facilitate the training step
% Ref: She, Xiwei, et al. Journal of Neuroscience Methods (2022)
%
% Author: Xiwei She, PhD
clear; clc;
addpath(genpath('toolbox'));

% Settings
runCase = 1; % 1: SR; 2: MR; 3: Shifted; 4: Shuffled

memoryLabelPool = 1:5; % Define Memory Label Here

nestedFold = 1:5;
resolution_all = [0:25, 50:5:100]; % Temporal resolution
num_split = 8; % Bagging splits
lambda_pool = power(exp(1), 0:-0.03:-9); % Lambda pool for regularization

for ca = memoryLabelPool

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
    for currentFold = nestedFold
        printStr1 = ['case ', mat2str(runCase), ' & ', Category, ' part#', mat2str(currentFold), ' Begine!'];
        disp(printStr1);

        parMD = MD_baselearner(Category, currentFold, runCase,...
            'par',1,'resolution_all',resolution_all,'num_split',num_split, 'lambda_pool', lambda_pool);

        parMD_R = parMD.run;
        printStr2 = ['case ', mat2str(runCase), ' & ', Category, ' part#', mat2str(currentFold), ' Done!'];
        disp(printStr2);
    end
end