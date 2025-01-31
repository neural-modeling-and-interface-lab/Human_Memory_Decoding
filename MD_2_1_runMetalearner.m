%% This is a running script for training (the second layer) meta-learner
% in the double-layer model using the class "MD_metalearner"
%
% It utilizes parallel computing strategy to facilitate the training step
% Ref: She, Xiwei, et al. Journal of Neuroscience Methods (2022)
%
% Author: Xiwei She, PhD
clear; clc;
addpath(genpath('toolbox'));
addpath('..');  %adds MemoryDecode Class and process_options

% Settings
nestedFold = 1:5;
num_split = 8;
lambda_pool = power(exp(1), 0:-0.03:-9); % Define the lambda pool

runCase = 1; % 1: SR; 2: MR; 3: Shifted; 4: Shuffle

categoryPool = 1:5;

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
    for currentFold = nestedFold
        printStr1 = ['case ', mat2str(runCase), ' & ', Category, ' part#', mat2str(currentFold), ' Begine!'];
        disp(printStr1);

        parMD = MD_metalearner(Category, currentFold, runCase,...
            'par',1,'num_split',num_split, 'lambda_pool', lambda_pool);

        parMD_R = parMD.run;
        printStr2 = ['case ', mat2str(runCase), ' & ', Category, ' part#', mat2str(currentFold), ' Done!'];
        disp(printStr2);
    end
end