%% This script uses the model-based SCFM to make classificaiton
%
% Author: Xiwei She
clear;clc;

%% Define name here
thisCase = '1 Sample Response';

%% Load SCFMs
load(['result\MD_SCFM_', thisCase(3:end),'_Animal.mat']);
SCFM_AnimalMask = SCFM_Prob_final_vis;
load(['result\MD_SCFM_', thisCase(3:end),'_Building.mat']);
SCFM_BuildingMask = SCFM_Prob_final_vis;
load(['result\MD_SCFM_', thisCase(3:end),'_Plant.mat']);
SCFM_PlantMask = SCFM_Prob_final_vis;
load(['result\MD_SCFM_', thisCase(3:end),'_Tool.mat']);
SCFM_ToolMask = SCFM_Prob_final_vis;
load(['result\MD_SCFM_', thisCase(3:end),'_Vehicle.mat']);
SCFM_VehicleMask = SCFM_Prob_final_vis;

%% Load any spatio-temporal patterns
iF1 = strcat(['result\MD_baselearner_', thisCase(3:end),'_Animal_fold1.mat']);
load(iF1);
STP_Animal = MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :);
STP0_Animal_std = std(sum(sum(MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :), 2), 3));

iF1 = strcat(['result\MD_baselearner_', thisCase(3:end),'_Building_fold1.mat']);
load(iF1);
STP_Building = MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :);
STP0_Building_std = std(sum(sum(MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :), 2), 3));

iF1 = strcat(['result\MD_baselearner_', thisCase(3:end),'_Plant_fold1.mat']);
load(iF1);
STP_Plant = MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :);
STP0_Plant_std = std(sum(sum(MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :), 2), 3));

iF1 = strcat(['result\MD_baselearner_', thisCase(3:end),'_Tool_fold1.mat']);
load(iF1);
STP_Tool = MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :);
STP0_Tool_std = std(sum(sum(MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :), 2), 3));

iF1 = strcat(['result\MD_baselearner_', thisCase(3:end),'_Vehicle_fold1.mat']);
load(iF1);
STP_Vehicle = MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :);
STP0_Vehicle_std = std(sum(sum(MD_baselearner.TrainingSet_SpikeTensor(MD_baselearner.TrainingSet_target==1, :, :), 2), 3));

%% Masked spatio-temporal patterns
% Without Masks
STP0_Animal = sum(sum(mean(STP_Animal, 1)));
STP0_Building = sum(sum(mean(STP_Building, 1)));
STP0_Plant = sum(sum(mean(STP_Plant, 1)));
STP0_Tool = sum(sum(mean(STP_Tool, 1)));
STP0_Vehicle = sum(sum(mean(STP_Vehicle, 1)));

% With Animal Mask
STP_AnimalMasked_Animal_trial = zeros(size(STP_Animal, 1), size(STP_Animal, 3), size(STP_Animal, 2));
STP_AnimalMasked_Building_trial = zeros(size(STP_Building, 1), size(STP_Building, 3), size(STP_Building, 2));
STP_AnimalMasked_Plant_trial = zeros(size(STP_Plant, 1), size(STP_Plant, 3), size(STP_Plant, 2));
STP_AnimalMasked_Tool_trial = zeros(size(STP_Tool, 1), size(STP_Tool, 3), size(STP_Tool, 2));
STP_AnimalMasked_Vehicle_trial = zeros(size(STP_Vehicle, 1), size(STP_Vehicle, 3), size(STP_Vehicle, 2));
for i = 1:size(STP_Animal, 1)
    STP_AnimalMasked_Animal_trial(i, :, :) = squeeze(STP_Animal(i, :, :))' .* SCFM_AnimalMask;
end
for i = 1:size(STP_Building, 1)
    STP_AnimalMasked_Building_trial(i, :, :) = squeeze(STP_Building(i, :, :))' .* SCFM_AnimalMask;
end
for i = 1:size(STP_Plant, 1)
    STP_AnimalMasked_Plant_trial(i, :, :) = squeeze(STP_Plant(i, :, :))' .* SCFM_AnimalMask;
end
for i = 1:size(STP_Tool, 1)
    STP_AnimalMasked_Tool_trial(i, :, :) = squeeze(STP_Tool(i, :, :))' .* SCFM_AnimalMask;
end
for i = 1:size(STP_Vehicle, 1)
    STP_AnimalMasked_Vehicle_trial(i, :, :) = squeeze(STP_Vehicle(i, :, :))' .* SCFM_AnimalMask;
end
STP_AnimalMasked_Animal = squeeze(mean(STP_AnimalMasked_Animal_trial, 1));
STP_AnimalMasked_Animal_std = std(sum(sum(STP_AnimalMasked_Animal_trial,3),2));
STP_AnimalMasked_Building = squeeze(mean(STP_AnimalMasked_Building_trial, 1));
STP_AnimalMasked_Building_std = std(sum(sum(STP_AnimalMasked_Building_trial,3),2));
STP_AnimalMasked_Plant = squeeze(mean(STP_AnimalMasked_Plant_trial, 1));
STP_AnimalMasked_Plant_std = std(sum(sum(STP_AnimalMasked_Plant_trial,3),2));
STP_AnimalMasked_Tool = squeeze(mean(STP_AnimalMasked_Tool_trial, 1));
STP_AnimalMasked_Tool_std = std(sum(sum(STP_AnimalMasked_Tool_trial,3),2));
STP_AnimalMasked_Vehicle = squeeze(mean(STP_AnimalMasked_Vehicle_trial, 1));
STP_AnimalMasked_Vehicle_std = std(sum(sum(STP_AnimalMasked_Vehicle_trial,3),2));

% With Building Mask
STP_BuildingMasked_Animal_trial = zeros(size(STP_Animal, 1), size(STP_Animal, 3), size(STP_Animal, 2));
STP_BuildingMasked_Building_trial = zeros(size(STP_Building, 1), size(STP_Building, 3), size(STP_Building, 2));
STP_BuildingMasked_Plant_trial = zeros(size(STP_Plant, 1), size(STP_Plant, 3), size(STP_Plant, 2));
STP_BuildingMasked_Tool_trial = zeros(size(STP_Tool, 1), size(STP_Tool, 3), size(STP_Tool, 2));
STP_BuildingMasked_Vehicle_trial = zeros(size(STP_Vehicle, 1), size(STP_Vehicle, 3), size(STP_Vehicle, 2));
for i = 1:size(STP_Animal, 1)
    STP_BuildingMasked_Animal_trial(i, :, :) = squeeze(STP_Animal(i, :, :))' .* SCFM_BuildingMask;
end
for i = 1:size(STP_Building, 1)
    STP_BuildingMasked_Building_trial(i, :, :) = squeeze(STP_Building(i, :, :))' .* SCFM_BuildingMask;
end
for i = 1:size(STP_Plant, 1)
    STP_BuildingMasked_Plant_trial(i, :, :) = squeeze(STP_Plant(i, :, :))' .* SCFM_BuildingMask;
end
for i = 1:size(STP_Tool, 1)
    STP_BuildingMasked_Tool_trial(i, :, :) = squeeze(STP_Tool(i, :, :))' .* SCFM_BuildingMask;
end
for i = 1:size(STP_Vehicle, 1)
    STP_BuildingMasked_Vehicle_trial(i, :, :) = squeeze(STP_Vehicle(i, :, :))' .* SCFM_BuildingMask;
end
STP_BuildingMasked_Animal = squeeze(mean(STP_BuildingMasked_Animal_trial, 1));
STP_BuildingMasked_Animal_std = std(sum(sum(STP_BuildingMasked_Animal_trial,3),2));
STP_BuildingMasked_Building = squeeze(mean(STP_BuildingMasked_Building_trial, 1));
STP_BuildingMasked_Building_std = std(sum(sum(STP_BuildingMasked_Building_trial,3),2));
STP_BuildingMasked_Plant = squeeze(mean(STP_BuildingMasked_Plant_trial, 1));
STP_BuildingMasked_Plant_std = std(sum(sum(STP_BuildingMasked_Plant_trial,3),2));
STP_BuildingMasked_Tool = squeeze(mean(STP_BuildingMasked_Tool_trial, 1));
STP_BuildingMasked_Tool_std = std(sum(sum(STP_BuildingMasked_Tool_trial,3),2));
STP_BuildingMasked_Vehicle = squeeze(mean(STP_BuildingMasked_Vehicle_trial, 1));
STP_BuildingMasked_Vehicle_std = std(sum(sum(STP_BuildingMasked_Vehicle_trial,3),2));

% With Plant Mask
STP_PlantMasked_Animal_trial = zeros(size(STP_Animal, 1), size(STP_Animal, 3), size(STP_Animal, 2));
STP_PlantMasked_Building_trial = zeros(size(STP_Building, 1), size(STP_Building, 3), size(STP_Building, 2));
STP_PlantMasked_Plant_trial = zeros(size(STP_Plant, 1), size(STP_Plant, 3), size(STP_Plant, 2));
STP_PlantMasked_Tool_trial = zeros(size(STP_Tool, 1), size(STP_Tool, 3), size(STP_Tool, 2));
STP_PlantMasked_Vehicle_trial = zeros(size(STP_Vehicle, 1), size(STP_Vehicle, 3), size(STP_Vehicle, 2));
for i = 1:size(STP_Animal, 1)
    STP_PlantMasked_Animal_trial(i, :, :) = squeeze(STP_Animal(i, :, :))' .* SCFM_PlantMask;
end
for i = 1:size(STP_Building, 1)
    STP_PlantMasked_Building_trial(i, :, :) = squeeze(STP_Building(i, :, :))' .* SCFM_PlantMask;
end
for i = 1:size(STP_Plant, 1)
    STP_PlantMasked_Plant_trial(i, :, :) = squeeze(STP_Plant(i, :, :))' .* SCFM_PlantMask;
end
for i = 1:size(STP_Tool, 1)
    STP_PlantMasked_Tool_trial(i, :, :) = squeeze(STP_Tool(i, :, :))' .* SCFM_PlantMask;
end
for i = 1:size(STP_Vehicle, 1)
    STP_PlantMasked_Vehicle_trial(i, :, :) = squeeze(STP_Vehicle(i, :, :))' .* SCFM_PlantMask;
end
STP_PlantMasked_Animal = squeeze(mean(STP_PlantMasked_Animal_trial, 1));
STP_PlantMasked_Animal_std = std(sum(sum(STP_PlantMasked_Animal_trial,3),2));
STP_PlantMasked_Building = squeeze(mean(STP_PlantMasked_Building_trial, 1));
STP_PlantMasked_Building_std = std(sum(sum(STP_PlantMasked_Building_trial,3),2));
STP_PlantMasked_Plant = squeeze(mean(STP_PlantMasked_Plant_trial, 1));
STP_PlantMasked_Plant_std = std(sum(sum(STP_PlantMasked_Plant_trial,3),2));
STP_PlantMasked_Tool = squeeze(mean(STP_PlantMasked_Tool_trial, 1));
STP_PlantMasked_Tool_std = std(sum(sum(STP_PlantMasked_Tool_trial,3),2));
STP_PlantMasked_Vehicle = squeeze(mean(STP_PlantMasked_Vehicle_trial, 1));
STP_PlantMasked_Vehicle_std = std(sum(sum(STP_PlantMasked_Vehicle_trial,3),2));

% With Tool Mask
STP_ToolMasked_Animal_trial = zeros(size(STP_Animal, 1), size(STP_Animal, 3), size(STP_Animal, 2));
STP_ToolMasked_Building_trial = zeros(size(STP_Building, 1), size(STP_Building, 3), size(STP_Building, 2));
STP_ToolMasked_Plant_trial = zeros(size(STP_Plant, 1), size(STP_Plant, 3), size(STP_Plant, 2));
STP_ToolMasked_Tool_trial = zeros(size(STP_Tool, 1), size(STP_Tool, 3), size(STP_Tool, 2));
STP_ToolMasked_Vehicle_trial = zeros(size(STP_Vehicle, 1), size(STP_Vehicle, 3), size(STP_Vehicle, 2));
for i = 1:size(STP_Animal, 1)
    STP_ToolMasked_Animal_trial(i, :, :) = squeeze(STP_Animal(i, :, :))' .* SCFM_ToolMask;
end
for i = 1:size(STP_Building, 1)
    STP_ToolMasked_Building_trial(i, :, :) = squeeze(STP_Building(i, :, :))' .* SCFM_ToolMask;
end
for i = 1:size(STP_Plant, 1)
    STP_ToolMasked_Plant_trial(i, :, :) = squeeze(STP_Plant(i, :, :))' .* SCFM_ToolMask;
end
for i = 1:size(STP_Tool, 1)
    STP_ToolMasked_Tool_trial(i, :, :) = squeeze(STP_Tool(i, :, :))' .* SCFM_ToolMask;
end
for i = 1:size(STP_Vehicle, 1)
    STP_ToolMasked_Vehicle_trial(i, :, :) = squeeze(STP_Vehicle(i, :, :))' .* SCFM_ToolMask;
end
STP_ToolMasked_Animal = squeeze(mean(STP_ToolMasked_Animal_trial, 1));
STP_ToolMasked_Animal_std = std(sum(sum(STP_ToolMasked_Animal_trial,3),2));
STP_ToolMasked_Building = squeeze(mean(STP_ToolMasked_Building_trial, 1));
STP_ToolMasked_Building_std = std(sum(sum(STP_ToolMasked_Building_trial,3),2));
STP_ToolMasked_Plant = squeeze(mean(STP_ToolMasked_Plant_trial, 1));
STP_ToolMasked_Plant_std = std(sum(sum(STP_ToolMasked_Plant_trial,3),2));
STP_ToolMasked_Tool = squeeze(mean(STP_ToolMasked_Tool_trial, 1));
STP_ToolMasked_Tool_std = std(sum(sum(STP_ToolMasked_Tool_trial,3),2));
STP_ToolMasked_Vehicle = squeeze(mean(STP_ToolMasked_Vehicle_trial, 1));
STP_ToolMasked_Vehicle_std = std(sum(sum(STP_ToolMasked_Vehicle_trial,3),2));

% With Building Mask
STP_VehicleMasked_Animal_trial = zeros(size(STP_Animal, 1), size(STP_Animal, 3), size(STP_Animal, 2));
STP_VehicleMasked_Building_trial = zeros(size(STP_Building, 1), size(STP_Building, 3), size(STP_Building, 2));
STP_VehicleMasked_Plant_trial = zeros(size(STP_Plant, 1), size(STP_Plant, 3), size(STP_Plant, 2));
STP_VehicleMasked_Tool_trial = zeros(size(STP_Tool, 1), size(STP_Tool, 3), size(STP_Tool, 2));
STP_VehicleMasked_Vehicle_trial = zeros(size(STP_Vehicle, 1), size(STP_Vehicle, 3), size(STP_Vehicle, 2));
for i = 1:size(STP_Animal, 1)
    STP_VehicleMasked_Animal_trial(i, :, :) = squeeze(STP_Animal(i, :, :))' .* SCFM_VehicleMask;
end
for i = 1:size(STP_Building, 1)
    STP_VehicleMasked_Building_trial(i, :, :) = squeeze(STP_Building(i, :, :))' .* SCFM_VehicleMask;
end
for i = 1:size(STP_Plant, 1)
    STP_VehicleMasked_Plant_trial(i, :, :) = squeeze(STP_Plant(i, :, :))' .* SCFM_VehicleMask;
end
for i = 1:size(STP_Tool, 1)
    STP_VehicleMasked_Tool_trial(i, :, :) = squeeze(STP_Tool(i, :, :))' .* SCFM_VehicleMask;
end
for i = 1:size(STP_Vehicle, 1)
    STP_VehicleMasked_Vehicle_trial(i, :, :) = squeeze(STP_Vehicle(i, :, :))' .* SCFM_VehicleMask;
end
STP_VehicleMasked_Animal = squeeze(mean(STP_VehicleMasked_Animal_trial, 1));
STP_VehicleMasked_Animal_std = std(sum(sum(STP_VehicleMasked_Animal_trial,3),2));
STP_VehicleMasked_Building = squeeze(mean(STP_VehicleMasked_Building_trial, 1));
STP_VehicleMasked_Building_std = std(sum(sum(STP_VehicleMasked_Building_trial,3),2));
STP_VehicleMasked_Plant = squeeze(mean(STP_VehicleMasked_Plant_trial, 1));
STP_VehicleMasked_Plant_std = std(sum(sum(STP_VehicleMasked_Plant_trial,3),2));
STP_VehicleMasked_Tool = squeeze(mean(STP_VehicleMasked_Tool_trial, 1));
STP_VehicleMasked_Tool_std = std(sum(sum(STP_VehicleMasked_Tool_trial,3),2));
STP_VehicleMasked_Vehicle = squeeze(mean(STP_VehicleMasked_Vehicle_trial, 1));
STP_VehicleMasked_Vehicle_std = std(sum(sum(STP_VehicleMasked_Vehicle_trial,3),2));

%% Visualization
% Animal
figure('Position', [50, 50, 900, 800]);
subplot(2, 1, 1)
h1 = barwitherr([STP0_Animal_std, STP0_Building_std, STP0_Plant_std, STP0_Tool_std, STP0_Vehicle_std], [STP0_Animal, STP0_Building, STP0_Plant, STP0_Tool, STP0_Vehicle]);
h1.LineWidth = 4;
%title('# of Spikes (without SCFM)')
set(gca, 'xticklabel', [], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)
subplot(2, 1, 2)
h2 = barwitherr([STP_AnimalMasked_Animal_std, STP_AnimalMasked_Building_std, STP_AnimalMasked_Plant_std, STP_AnimalMasked_Tool_std, STP_AnimalMasked_Vehicle_std], [sum(STP_AnimalMasked_Animal(:)), sum(STP_AnimalMasked_Building(:)), sum(STP_AnimalMasked_Plant(:)), sum(STP_AnimalMasked_Tool(:)), sum(STP_AnimalMasked_Vehicle(:))], 'r');
h2.FaceColor = 'r';
h2.LineWidth = 4;
%title(['# of Spikes (with SCFM-A)'])
set(gca, 'xticklabel', ['A'; 'B'; 'P'; 'T'; 'V'], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)

% Building
figure('Position', [50, 50, 900, 800]);
subplot(2, 1, 1)
h1 = barwitherr([STP0_Animal_std, STP0_Building_std, STP0_Plant_std, STP0_Tool_std, STP0_Vehicle_std], [STP0_Animal, STP0_Building, STP0_Plant, STP0_Tool, STP0_Vehicle]);
% h1.FaceColor = 'b';
h1.LineWidth = 4;
%title('# of Spikes (without SCFM)')
set(gca, 'xticklabel', [], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)
subplot(2, 1, 2)
h2 = barwitherr([STP_BuildingMasked_Animal_std, STP_BuildingMasked_Building_std, STP_BuildingMasked_Plant_std, STP_BuildingMasked_Tool_std, STP_BuildingMasked_Vehicle_std], [sum(STP_BuildingMasked_Animal(:)), sum(STP_BuildingMasked_Building(:)), sum(STP_BuildingMasked_Plant(:)), sum(STP_BuildingMasked_Tool(:)), sum(STP_BuildingMasked_Vehicle(:))], 'r');
h2.FaceColor = 'r';
h2.LineWidth = 4;
%title(['# of Spikes (with SCFM-A)'])
set(gca, 'xticklabel', ['A'; 'B'; 'P'; 'T'; 'V'], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)

% Plant
figure('Position', [50, 50, 900, 800]);
subplot(2, 1, 1)
h1 = barwitherr([STP0_Animal_std, STP0_Building_std, STP0_Plant_std, STP0_Tool_std, STP0_Vehicle_std], [STP0_Animal, STP0_Building, STP0_Plant, STP0_Tool, STP0_Vehicle]);
% h1.FaceColor = 'b';
h1.LineWidth = 4;
%title('# of Spikes (without SCFM)')
set(gca, 'xticklabel', [], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)
subplot(2, 1, 2)
h2 = barwitherr([STP_PlantMasked_Animal_std, STP_PlantMasked_Building_std, STP_PlantMasked_Plant_std, STP_PlantMasked_Tool_std, STP_PlantMasked_Vehicle_std], [sum(STP_PlantMasked_Animal(:)), sum(STP_PlantMasked_Building(:)), sum(STP_PlantMasked_Plant(:)), sum(STP_PlantMasked_Tool(:)), sum(STP_PlantMasked_Vehicle(:))], 'r');
h2.FaceColor = 'r';
h2.LineWidth = 4;
%title(['# of Spikes (with SCFM-A)'])
set(gca, 'xticklabel', ['A'; 'B'; 'P'; 'T'; 'V'], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)

% Tool
figure('Position', [50, 50, 900, 800]);
subplot(2, 1, 1)
h1 = barwitherr([STP0_Animal_std, STP0_Building_std, STP0_Plant_std, STP0_Tool_std, STP0_Vehicle_std], [STP0_Animal, STP0_Building, STP0_Plant, STP0_Tool, STP0_Vehicle]);
% h1.FaceColor = 'b';
h1.LineWidth = 4;
%title('# of Spikes (without SCFM)')
set(gca, 'xticklabel', [], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)
subplot(2, 1, 2)
h2 = barwitherr([STP_ToolMasked_Animal_std, STP_ToolMasked_Building_std, STP_ToolMasked_Plant_std, STP_ToolMasked_Tool_std, STP_ToolMasked_Vehicle_std], [sum(STP_ToolMasked_Animal(:)), sum(STP_ToolMasked_Building(:)), sum(STP_ToolMasked_Plant(:)), sum(STP_ToolMasked_Tool(:)), sum(STP_ToolMasked_Vehicle(:))], 'r');
h2.FaceColor = 'r';
h2.LineWidth = 4;
%title(['# of Spikes (with SCFM-A)'])
set(gca, 'xticklabel', ['A'; 'B'; 'P'; 'T'; 'V'], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)

% Vehicle
figure('Position', [50, 50, 900, 800]);
subplot(2, 1, 1)
h1 = barwitherr([STP0_Animal_std, STP0_Building_std, STP0_Plant_std, STP0_Tool_std, STP0_Vehicle_std], [STP0_Animal, STP0_Building, STP0_Plant, STP0_Tool, STP0_Vehicle]);
% h1.FaceColor = 'b';
h1.LineWidth = 4;
%title('# of Spikes (without SCFM)')
set(gca, 'xticklabel', [], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)
subplot(2, 1, 2)
h2 = barwitherr([STP_VehicleMasked_Animal_std, STP_VehicleMasked_Building_std, STP_VehicleMasked_Plant_std, STP_VehicleMasked_Tool_std, STP_VehicleMasked_Vehicle_std], [sum(STP_VehicleMasked_Animal(:)), sum(STP_VehicleMasked_Building(:)), sum(STP_VehicleMasked_Plant(:)), sum(STP_VehicleMasked_Tool(:)), sum(STP_VehicleMasked_Vehicle(:))], 'r');
h2.FaceColor = 'r';
h2.LineWidth = 4;
%title(['# of Spikes (with SCFM-A)'])
set(gca, 'xticklabel', ['A'; 'B'; 'P'; 'T'; 'V'], 'yticklabel', [], 'FontName', 'Arial','FontWeight','normal', 'Fontsize', 40, 'linewidth', 4)