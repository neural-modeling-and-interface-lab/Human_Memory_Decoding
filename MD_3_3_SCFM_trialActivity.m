%% This script is used to visualize the histogrm of
% category-specific patterns between decoded trials vs. non-decoded trials
% for all categories
%
% Author: Xiwei She
clear;clc;
close all

addpath('toolbox');

runCase = 1; % 1: SR; 2: MR; 3: Shifted Control; 4: Shuffle Control

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

%% Load data
BehavioralData = load('exampleData\BehavioralData.mat');
NeuralData = load('exampleData\NeuralData.mat');
LabelData = load('exampleData\LabelData.mat');
ChannelInfo = load('exampleData\ChannelInfo.mat');

%% Load SCFM results
SCFMPath = 'result';

SCFM_Animal = load([SCFMPath, '\MD_SCFM_', thisCase(3:end),'_Animal.mat']);
SCFM_Animal = SCFM_Animal.SCFM_Prob_final_vis;
SCFM_Building = load([SCFMPath, '\MD_SCFM_', thisCase(3:end),'_Building.mat']);
SCFM_Building = SCFM_Building.SCFM_Prob_final_vis;
SCFM_Plant = load([SCFMPath, '\MD_SCFM_', thisCase(3:end),'_Plant.mat']);
SCFM_Plant = SCFM_Plant.SCFM_Prob_final_vis;
SCFM_Tool = load([SCFMPath, '\MD_SCFM_', thisCase(3:end),'_Tool.mat']);
SCFM_Tool = SCFM_Tool.SCFM_Prob_final_vis;
SCFM_Vehicle = load([SCFMPath, '\MD_SCFM_', thisCase(3:end),'_Vehicle.mat']);
SCFM_Vehicle = SCFM_Vehicle.SCFM_Prob_final_vis;

%% Epoching
L = 1000; % # of 2 msec bins in a pattern
SpikeTensor = Train2Tensor([NeuralData.X NeuralData.Y], BehavioralData.SAMPLE_RESPONSE, L);

%% Visualization

vis_neuron_index = 1; % Pick any neuron for visualization
nbins = 10;

currentNeuron = SpikeTensor(:, :, vis_neuron_index);

% Calculate baseline firing rate
currentNeuron_ave = mean(SpikeTensor(:, :, vis_neuron_index), 1);
baseFR = mean(sum(reshape(currentNeuron_ave(1:end-1), [], nbins), 1));

% Calculate Animal vs. Non-Animal firing rate
tempNeuronTensor_Animal = currentNeuron(LabelData.AveAnimal == 1, 1:L);
numTrial_Animal = size(tempNeuronTensor_Animal, 1);
tempNeuronTensor_Animal_binned = reshape(tempNeuronTensor_Animal, numTrial_Animal, L/nbins, nbins);
tempNeuronTensor_Animal_binned_summed = sum(squeeze(mean(tempNeuronTensor_Animal_binned, 1)));

tempNeuronTensor_NonAnimal = currentNeuron(LabelData.AveAnimal ~= 1, 1:L);
numTrial_NonAnimal = size(tempNeuronTensor_NonAnimal, 1);
tempNeuronTensor_NonAnimal_binned = reshape(tempNeuronTensor_NonAnimal, numTrial_NonAnimal, L/nbins, nbins);
tempNeuronTensor_NonAnimal_binned_summed = sum(squeeze(mean(tempNeuronTensor_NonAnimal_binned, 1)));

pValues_Animal = zeros(nbins, 1);
hValues_Animal = zeros(nbins, 1);
for bIndex = 1:nbins
    temp1 = tempNeuronTensor_Animal_binned(:, :, bIndex);
    temp2 = tempNeuronTensor_NonAnimal_binned(:, :, bIndex);

    [h, p] = ttest2(temp1(:), temp2(:), 'Alpha', 0.01);
    hValues_Animal(bIndex) = h;
    pValues_Animal(bIndex) = p;
end


% Calculate Building vs. Non-Building firing rate
tempNeuronTensor_Building = currentNeuron(LabelData.AveBuilding == 1, 1:L);
numTrial_Building = size(tempNeuronTensor_Building, 1);
tempNeuronTensor_Building_binned = reshape(tempNeuronTensor_Building, numTrial_Building, L/nbins, nbins);
tempNeuronTensor_Building_binned_summed = sum(squeeze(mean(tempNeuronTensor_Building_binned, 1)));

tempNeuronTensor_NonBuilding = currentNeuron(LabelData.AveBuilding ~= 1, 1:L);
numTrial_NonBuilding = size(tempNeuronTensor_NonBuilding, 1);
tempNeuronTensor_NonBuilding_binned = reshape(tempNeuronTensor_NonBuilding, numTrial_NonBuilding, L/nbins, nbins);
tempNeuronTensor_NonBuilding_binned_summed = sum(squeeze(mean(tempNeuronTensor_NonBuilding_binned, 1)));

pValues_Building = zeros(nbins, 1);
hValues_Building = zeros(nbins, 1);
for bIndex = 1:nbins
    temp1 = tempNeuronTensor_Building_binned(:, :, bIndex);
    temp2 = tempNeuronTensor_NonBuilding_binned(:, :, bIndex);

    [h, p] = ttest2(temp1(:), temp2(:), 'Alpha', 0.01);
    hValues_Building(bIndex) = h;
    pValues_Building(bIndex) = p;
end

% Calculate Plant vs. Non-Plant firing rate
tempNeuronTensor_Plant = currentNeuron(LabelData.AvePlant == 1, 1:L);
numTrial_Plant = size(tempNeuronTensor_Plant, 1);
tempNeuronTensor_Plant_binned = reshape(tempNeuronTensor_Plant, numTrial_Plant, L/nbins, nbins);
tempNeuronTensor_Plant_binned_summed = sum(squeeze(mean(tempNeuronTensor_Plant_binned, 1)));

tempNeuronTensor_NonPlant = currentNeuron(LabelData.AvePlant ~= 1, 1:L);
numTrial_NonPlant = size(tempNeuronTensor_NonPlant, 1);
tempNeuronTensor_NonPlant_binned = reshape(tempNeuronTensor_NonPlant, numTrial_NonPlant, L/nbins, nbins);
tempNeuronTensor_NonPlant_binned_summed = sum(squeeze(mean(tempNeuronTensor_NonPlant_binned, 1)));

pValues_Plant = zeros(nbins, 1);
hValues_Plant = zeros(nbins, 1);
for bIndex = 1:nbins
    temp1 = tempNeuronTensor_Plant_binned(:, :, bIndex);
    temp2 = tempNeuronTensor_NonPlant_binned(:, :, bIndex);

    [h, p] = ttest2(temp1(:), temp2(:), 'Alpha', 0.01);
    hValues_Plant(bIndex) = h;
    pValues_Plant(bIndex) = p;
end

% Calculate Tool vs. Non-Tool firing rate
tempNeuronTensor_Tool = currentNeuron(LabelData.AveTool == 1, 1:L);
numTrial_Tool = size(tempNeuronTensor_Tool, 1);
tempNeuronTensor_Tool_binned = reshape(tempNeuronTensor_Tool, numTrial_Tool, L/nbins, nbins);
tempNeuronTensor_Tool_binned_summed = sum(squeeze(mean(tempNeuronTensor_Tool_binned, 1)));

tempNeuronTensor_NonTool = currentNeuron(LabelData.AveTool ~= 1, 1:L);
numTrial_NonTool = size(tempNeuronTensor_NonTool, 1);
tempNeuronTensor_NonTool_binned = reshape(tempNeuronTensor_NonTool, numTrial_NonTool, L/nbins, nbins);
tempNeuronTensor_NonTool_binned_summed = sum(squeeze(mean(tempNeuronTensor_NonTool_binned, 1)));

pValues_Tool = zeros(nbins, 1);
hValues_Tool = zeros(nbins, 1);
for bIndex = 1:nbins
    temp1 = tempNeuronTensor_Tool_binned(:, :, bIndex);
    temp2 = tempNeuronTensor_NonTool_binned(:, :, bIndex);

    [h, p] = ttest2(temp1(:), temp2(:), 'Alpha', 0.01);
    hValues_Tool(bIndex) = h;
    pValues_Tool(bIndex) = p;
end

% Calculate Vehicle vs. Non-Vehicle firing rate
tempNeuronTensor_Vehicle = currentNeuron(LabelData.AveVehicle == 1, 1:L);
numTrial_Vehicle = size(tempNeuronTensor_Vehicle, 1);
tempNeuronTensor_Vehicle_binned = reshape(tempNeuronTensor_Vehicle, numTrial_Vehicle, L/nbins, nbins);
tempNeuronTensor_Vehicle_binned_summed = sum(squeeze(mean(tempNeuronTensor_Vehicle_binned, 1)));

tempNeuronTensor_NonVehicle = currentNeuron(LabelData.AveVehicle ~= 1, 1:L);
numTrial_NonVehicle = size(tempNeuronTensor_NonVehicle, 1);
tempNeuronTensor_NonVehicle_binned = reshape(tempNeuronTensor_NonVehicle, numTrial_NonVehicle, L/nbins, nbins);
tempNeuronTensor_NonVehicle_binned_summed = sum(squeeze(mean(tempNeuronTensor_NonVehicle_binned, 1)));


%% Statistic Comparison
pValues_Vehicle = zeros(nbins, 1);
hValues_Vehicle = zeros(nbins, 1);
for bIndex = 1:nbins
    temp1 = tempNeuronTensor_Vehicle_binned(:, :, bIndex);
    temp2 = tempNeuronTensor_NonVehicle_binned(:, :, bIndex);

    [h, p] = ttest2(temp1(:), temp2(:), 'Alpha', 0.01);
    hValues_Vehicle(bIndex) = h;
    pValues_Vehicle(bIndex) = p;
end

% Visualization
yRange = max([tempNeuronTensor_Animal_binned_summed, tempNeuronTensor_Building_binned_summed, tempNeuronTensor_Plant_binned_summed, tempNeuronTensor_Tool_binned_summed, tempNeuronTensor_Vehicle_binned_summed, ...
    tempNeuronTensor_NonAnimal_binned_summed, tempNeuronTensor_NonBuilding_binned_summed, tempNeuronTensor_NonPlant_binned_summed, tempNeuronTensor_NonTool_binned_summed, tempNeuronTensor_NonVehicle_binned_summed]);

figure('Position', [50, 10, 1700, 900])
tiledlayout(3,5, 'Padding', 'none', 'TileSpacing', 'compact');

%% SCFM
nexttile
h_Animal = heatmap(flip(SCFM_Animal),'GridVisible','off'); % Put the first neuron at bottom
h_Animal.Colormap = redwhiteblue(min(min(SCFM_Animal)), max(max(SCFM_Animal)));
h_Animal.XDisplayLabels = nan(size(h_Animal.XDisplayData));
h_Animal.YDisplayLabels = nan(size(h_Animal.YDisplayData));
h_Animal.ColorbarVisible = 'off'; 
ylabel('Neurons'); title('Animal')

nexttile
h_Building = heatmap(flip(SCFM_Building),'GridVisible','off'); % Put the first neuron at bottom
h_Building.Colormap = redwhiteblue(min(min(SCFM_Building)), max(max(SCFM_Building)));
h_Building.XDisplayLabels = nan(size(h_Building.XDisplayData));
h_Building.YDisplayLabels = nan(size(h_Building.YDisplayData));
h_Building.ColorbarVisible = 'off';
title('Building')

nexttile
h_Plant = heatmap(flip(SCFM_Plant),'GridVisible','off'); % Put the first neuron at bottom
h_Plant.Colormap = redwhiteblue(min(min(SCFM_Plant)), max(max(SCFM_Plant)));
h_Plant.XDisplayLabels = nan(size(h_Plant.XDisplayData));
h_Plant.YDisplayLabels = nan(size(h_Plant.YDisplayData));
h_Plant.ColorbarVisible = 'off';
title('Plant')

nexttile
h_Tool = heatmap(flip(SCFM_Tool),'GridVisible','off'); % Put the first neuron at bottom
h_Tool.Colormap = redwhiteblue(min(min(SCFM_Tool)), max(max(SCFM_Tool)));
h_Tool.XDisplayLabels = nan(size(h_Tool.XDisplayData));
h_Tool.YDisplayLabels = nan(size(h_Tool.YDisplayData));
h_Tool.ColorbarVisible = 'off';
title('Tool')

nexttile
h_Vehicle = heatmap(flip(SCFM_Vehicle),'GridVisible','off'); % Put the first neuron at bottom
h_Vehicle.Colormap = redwhiteblue(min(min(SCFM_Vehicle)), max(max(SCFM_Vehicle)));
h_Vehicle.XDisplayLabels = nan(size(h_Vehicle.XDisplayData));
h_Vehicle.YDisplayLabels = nan(size(h_Vehicle.YDisplayData));
h_Vehicle.ColorbarVisible = 'off';
title('Vehicle')

%% Decoded Trials

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on;
for bIndex = 1:nbins
    if hValues_Animal(bIndex) == 1
        plot(bIndex, yRange, '*k'); hold on
    end
end
bar(tempNeuronTensor_Animal_binned_summed, 'BarWidth', 1, 'FaceColor', [0.8500 0.3250 0.0980])
ylim([0, yRange + 0.1]); title('Animal')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
for bIndex = 1:nbins
    if hValues_Building(bIndex) == 1
        plot(bIndex, yRange, '*k'); hold on
    end
end
bar(tempNeuronTensor_Building_binned_summed, 'BarWidth', 1, 'FaceColor', [0.8500 0.3250 0.0980])
ylim([0, yRange + 0.1]); yticklabels({''}); title('Building')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
for bIndex = 1:nbins
    if hValues_Plant(bIndex) == 1
        plot(bIndex, yRange, '*k'); hold on
    end
end
bar(tempNeuronTensor_Plant_binned_summed, 'BarWidth', 1, 'FaceColor', [0.8500 0.3250 0.0980])
ylim([0, yRange + 0.1]); yticklabels({''}); title('Plant')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
for bIndex = 1:nbins
    if hValues_Tool(bIndex) == 1
        plot(bIndex, yRange, '*k'); hold on
    end
end
bar(tempNeuronTensor_Tool_binned_summed, 'BarWidth', 1, 'FaceColor', [0.8500 0.3250 0.0980])
ylim([0, yRange + 0.1]); yticklabels({''}); title('Tool')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
for bIndex = 1:nbins
    if hValues_Vehicle(bIndex) == 1
        plot(bIndex, yRange, '*k'); hold on
    end
end
bar(tempNeuronTensor_Vehicle_binned_summed, 'BarWidth', 1, 'FaceColor', [0.8500 0.3250 0.0980])
ylim([0, yRange + 0.1]); yticklabels({''}); title('Vehicle')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

%% Non-decoded trials
nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
bar(tempNeuronTensor_NonAnimal_binned_summed, 'BarWidth', 1)
ylim([0, yRange + 0.1]); title('Non-Animal')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
bar(tempNeuronTensor_NonBuilding_binned_summed, 'BarWidth', 1)
ylim([0, yRange + 0.1]); yticklabels({''}); title('Non-Building')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
bar(tempNeuronTensor_NonPlant_binned_summed, 'BarWidth', 1)
ylim([0, yRange + 0.1]); yticklabels({''}); title('Non-Plant')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
bar(tempNeuronTensor_NonTool_binned_summed, 'BarWidth', 1)
ylim([0, yRange + 0.1]); yticklabels({''}); title('Non-Tool')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

nexttile
yline(baseFR, '--', 'LineWidth', 3); hold on
bar(tempNeuronTensor_NonVehicle_binned_summed, 'BarWidth', 1)
ylim([0, yRange + 0.1]); yticklabels({''}); title('Non-Vehicle')
xticks([0.5, 1.5, 2.5, 3, 3.5, 4.5, 5.5, 6.5, 7.5, 8, 8.5, 9.5, 10.5]);
xticklabels({'-1', '', '', '', '', '', '0', '', '', '', '', '', '1'})
set(gca, 'FontName', 'Arial', 'FontSize', 20)

