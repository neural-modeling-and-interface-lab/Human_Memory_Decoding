%% This script is used to generate input-output datasets for training
% the double-layer models
% It also examines the data properties such as firing rate, autocorrelogram
%
% Author: Xiwei She, PhD

clear; clc;
addpath(genpath('toolbox'));

%% Define cases
runCase = 1; % 1: SR; 2: MR; 3: Shifted; 4: Shuffled

categoryPool = 1:5; % 1:Animal; 2:Building; 3:Plant; 4:Tool; 5:Vehicle

%% Load Data
iF1 = strcat('exampleData\BehavioralData.mat');
load(iF1, 'SAMPLE_ON', 'SAMPLE_RESPONSE', 'MATCH_RESPONSE');

iF2 = strcat('exampleData\NeuralData.mat');
load(iF2, 'X', 'Y');

iF3 = strcat('exampleData\LabelData.mat');
load(iF3, 'AveAnimal', 'AveBuilding', 'AvePlant', 'AveTool', 'AveVehicle');

%% Examine basic neural information

L = 1000; % Decoding window sidze. 2 seconds e.g., [-1 to 1] to SR/MR

binsize = 2; %specified in ms
numCA3 = size(X, 2);
numCA1 = size(Y, 2);
dataLen = size(X, 1); % Recording Length

CA3FR = zeros(numCA3, 1); % CA3 neuron firing rate
for n = 1:(numCA3)
    CA3FR(n) = sum(full(X(:, n)))  * 1000 / binsize / dataLen;
end
badChan = CA3FR<1 | CA3FR>20;
X(:, badChan) = []; % Remove bad channels
CA3FR(badChan) = [];
numCA3 = size(X, 2);

CA1FR = zeros(numCA1, 1); % CA1 neuron firing rate
for n = 1:(numCA1)
    CA1FR(n) = sum(full(Y(:, n)))  * 1000 / binsize / dataLen;
end
badChan = find(CA1FR<1 | CA1FR>20);
Y(:, badChan) = []; % Remove bad channels
CA1FR(badChan) = [];
numCA1 = size(Y, 2);

disp(['Number of CA3 neurons: ', mat2str(numCA3), ' & number of CA1 neurons: ', mat2str(numCA1)])

aveFR = (mean(CA3FR) + mean(CA1FR)) / 2; % Average firing rate
disp(['Average firing rate of this dataset is: ', mat2str(aveFR)])
disp(' ')

%% Generate Input-output data
switch runCase
    case 1
        decodingWindow = SAMPLE_RESPONSE;
        caseName = '1 Sample Response';
    case 2
        decodingWindow = MATCH_RESPONSE;
        caseName = '2 Match Response';
    case 3
        decodingWindow = SAMPLE_ON - L/1000;
        negativeIndex = find(decodingWindow < 0); % The first trial
        decodingWindow(negativeIndex) = L/1000 + 0.01;
        caseName = '3 Shifted Control';
    case 4
        decodingWindow = SAMPLE_RESPONSE;
        caseName = '4 Shuffle Control';
end

seed = 1; rng(seed); % For reproducibility
for ca = categoryPool

    switch ca
        case 1 
            Category = 'Animal';
            target = AveAnimal;
        case 2 
            Category = 'Building';
            target = AveBuilding;
        case 3 
            Category = 'Plant';
            target = AvePlant;
        case 4 
            Category = 'Tool';
            target = AveTool;
        case 5 
            Category = 'Vehicle';
            target = AveVehicle;
    end

    % Generate input tensor: Trial * Time * Neuron
    SpikeTensor = Train2Tensor([X Y], decodingWindow, L);

    % Uncomment to check autocorrelogram for all neurons
%     vis_autocorrelogram(SpikeTensor);

    % Divide data into 10 / 5 sets for nested CV
    partitionFolds = 5; % 10 / 5
    CrossValSet = cvpartition(length(target),'KFold', partitionFolds);

    for partition = 1:partitionFolds
        TrainingSet_SpikeTensor = SpikeTensor(training(CrossValSet, partition),:, :);
        TrainingSet_target = target(training(CrossValSet, partition),:, :);
        TestingSet_SpikeTensor = SpikeTensor(test(CrossValSet, partition),:, :);
        TestingSet_target = target(test(CrossValSet, partition),:, :);

        % Set Output File
        oF = strcat('processedData\', caseName,'\MD_',Category,'_split',num2str(partition),'.mat');
        save(oF, 'TrainingSet_SpikeTensor', 'TrainingSet_target', 'TestingSet_SpikeTensor', 'TestingSet_target', 'target', 'SpikeTensor', 'CrossValSet')
    end

end