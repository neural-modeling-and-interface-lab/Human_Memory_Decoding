%% This script visualize the performance of MD modeling by
% using a combine visualization
% of violin plots and pentagon plots
%
% Author: Xiwei She

clear; clc;
addpath(genpath('toolbox'));
warning off

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

%% Get all resutls
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

    % Load Results
    iF = strcat('result\MD_metalearner_',thisCase(3:end), '_', Category,'_Performance.mat');
    load(iF, 'bestFirstLayerMCC_training', 'bestFirstLayerMCC_testing', 'bestSecondLayerMCC_training', 'bestSecondLayerMCC_testing');

    MCC_1_overall(1, ca) = bestFirstLayerMCC_testing;
    MCC_2_overall(1, ca) = bestSecondLayerMCC_testing;
end


%% Get violin distribution
direction = [0, 0, 1];
widthScale = 0.5;
h2 = figure(2);
[vh_animal, ~, mean_animal, ~, ~] = violin(MCC_2_overall(:, 1), 'facecolor', [1, 0, 0]);
violin_animal_x = vh_animal.XData - 1;
violin_animal_x = violin_animal_x * widthScale;
violin_animal_y = vh_animal.YData;
violin_animal_y = mapminmax(violin_animal_y', 0, max(MCC_2_overall(:, 1)))';
violin_animal_mean_x = [min(violin_animal_x) max(violin_animal_x)];
violin_animal_mean_y = mean_animal;

[vh_building, ~, mean_building, ~, ~] = violin(MCC_2_overall(:, 2), 'facecolor', [0, 1, 1]);
violin_building_x = vh_building.XData - 1;
violin_building_x = violin_building_x * widthScale;
violin_building_y = vh_building.YData;
violin_building_y = mapminmax(violin_building_y', 0, max(MCC_2_overall(:, 2)))';
violin_building_mean_x = [min(violin_building_x) max(violin_building_x)];
violin_building_mean_y = mean_building;

[vh_plant, ~, mean_plant, ~, ~] = violin(MCC_2_overall(:, 3), 'facecolor', [0, 1, 0]);
violin_plant_x = vh_plant.XData - 1;
violin_plant_x = violin_plant_x * widthScale;
violin_plant_y = vh_plant.YData;
violin_plant_y = mapminmax(violin_plant_y', 0, max(MCC_2_overall(:, 3)))';
violin_plant_mean_x = [min(violin_plant_x) max(violin_plant_x)];
violin_plant_mean_y = mean_plant;

[vh_tool, ~, mean_tool, ~, ~] = violin(MCC_2_overall(:, 4), 'facecolor', [1, 1, 0]);
violin_tool_x = vh_tool.XData - 1;
violin_tool_x = violin_tool_x * widthScale;
violin_tool_y = vh_tool.YData;
violin_tool_y = mapminmax(violin_tool_y', 0, max(MCC_2_overall(:, 4)))';
violin_tool_mean_x = [min(violin_tool_x) max(violin_tool_x)];
violin_tool_mean_y = mean_tool;

[vh_vehicle, ~, mean_vehicle, ~, ~] = violin(MCC_2_overall(:, 5), 'facecolor', [0.5, 0.5, 1]);
violin_vehicle_x = vh_vehicle.XData - 1;
violin_vehicle_x = violin_vehicle_x * widthScale;
violin_vehicle_y = vh_vehicle.YData;
violin_vehicle_y = mapminmax(violin_vehicle_y', 0, max(MCC_2_overall(:, 5)))';
violin_vehicle_mean_x = [min(violin_vehicle_x) max(violin_vehicle_x)];
violin_vehicle_mean_y = mean_vehicle;

close figure 2

%% Overall visualization
Xc = 0; Yc = 0; % Pentagon center
h1 = figure('position', [50, 50, 1050, 950]);
% Formate the pentagon outline
p1 = nsidedpoly(5, 'Center', [0 0], 'Radius', 0.86);
h1 = plot(p1); hold on;
h1.FaceColor = [0.2 0.2 0.2];
h1.EdgeColor = [1 1 1];
h1.LineWidth = 1;
set(gca, 'box', 'off', 'xtick', [], 'ytick', [], 'fontSize', 24, 'fontName', 'Arial')
xVector = h1.Shape.Vertices(:, 1);
yVector = h1.Shape.Vertices(:, 2);

% Plot annotations
dim_Animal = [0.27 0.16 .1 .05];
str = 'Animal';
t = annotation('textbox',dim_Animal,'String',str, 'fontSize', 20); hold on;
t.EdgeColor = [1 1 1];

dim_Building = [0.14 0.65 .1 .05];
str = 'Building';
t = annotation('textbox',dim_Building,'String',str, 'fontSize', 20); hold on;
t.EdgeColor = [1 1 1];

dim_Plant = [0.48 0.86 .1 .05];
str = 'Plant';
t = annotation('textbox',dim_Plant,'String',str, 'fontSize', 20); hold on;
t.EdgeColor = [1 1 1];

dim_Tool = [0.82 0.65 .1 .05];
str = 'Tool';
t = annotation('textbox',dim_Tool,'String',str, 'fontSize', 20); hold on;
t.EdgeColor = [1 1 1];

dim_Vehicle = [0.67 0.16 .1 .05];
str = 'Vehicle';
t = annotation('textbox',dim_Vehicle,'String',str, 'fontSize', 20); hold on;
t.EdgeColor = [1 1 1];

ang_animal = -40.21;
violin_animal_x_2 =  (violin_animal_x-Xc)*cos(ang_animal) + (violin_animal_y-Yc)*sin(ang_animal) + Xc;
violin_animal_y_2 = -(violin_animal_x-Xc)*sin(ang_animal) + (violin_animal_y-Yc)*cos(ang_animal) + Yc;
fill(violin_animal_x_2, violin_animal_y_2, 'w','LineStyle','none'); hold on;
violin_animal_mean_x_2 =  (violin_animal_mean_x-Xc)*cos(ang_animal) + (violin_animal_mean_y-Yc)*sin(ang_animal) + Xc;
violin_animal_mean_y_2 = -(violin_animal_mean_x-Xc)*sin(ang_animal) + (violin_animal_mean_y-Yc)*cos(ang_animal) + Yc;
violin_animal_mean_center = [mean(violin_animal_mean_x_2), mean(violin_animal_mean_y_2)];
% plot(violin_animal_mean_x_2, violin_animal_mean_y_2, 'k', 'LineWidth', 2.5); hold on;

ang_building = -20.1;
violin_building_x_2 =  (violin_building_x-Xc)*cos(ang_building) + (violin_building_y-Yc)*sin(ang_building) + Xc;
violin_building_y_2 = -(violin_building_x-Xc)*sin(ang_building) + (violin_building_y-Yc)*cos(ang_building) + Yc;
fill(violin_building_x_2, violin_building_y_2, 'w','LineStyle','none'); hold on;
violin_building_mean_x_2 =  (violin_building_mean_x-Xc)*cos(ang_building) + (violin_building_mean_y-Yc)*sin(ang_building) + Xc;
violin_building_mean_y_2 = -(violin_building_mean_x-Xc)*sin(ang_building) + (violin_building_mean_y-Yc)*cos(ang_building) + Yc;
violin_building_mean_center = [mean(violin_building_mean_x_2), mean(violin_building_mean_y_2)];
% plot(violin_building_mean_x_2, violin_building_mean_y_2, 'k', 'LineWidth', 2.5); hold on;

ang_plant = 0;
violin_plant_x_2 =  (violin_plant_x-Xc)*cos(ang_plant) + (violin_plant_y-Yc)*sin(ang_plant) + Xc;
violin_plant_y_2 = -(violin_plant_x-Xc)*sin(ang_plant) + (violin_plant_y-Yc)*cos(ang_plant) + Yc;
fill(violin_plant_x_2, violin_plant_y_2, 'w','LineStyle','none'); hold on;
violin_plant_mean_x_2 =  (violin_plant_mean_x-Xc)*cos(ang_plant) + (violin_plant_mean_y-Yc)*sin(ang_plant) + Xc;
violin_plant_mean_y_2 = -(violin_plant_mean_x-Xc)*sin(ang_plant) + (violin_plant_mean_y-Yc)*cos(ang_plant) + Yc;
violin_plant_mean_center = [mean(violin_plant_mean_x_2), mean(violin_plant_mean_y_2)];
% plot(violin_plant_mean_x_2, violin_plant_mean_y_2, 'k', 'LineWidth', 2.5); hold on;

ang_tool = 20.1;
violin_tool_x_2 =  (violin_tool_x-Xc)*cos(ang_tool) + (violin_tool_y-Yc)*sin(ang_tool) + Xc;
violin_tool_y_2 = -(violin_tool_x-Xc)*sin(ang_tool) + (violin_tool_y-Yc)*cos(ang_tool) + Yc;
fill(violin_tool_x_2, violin_tool_y_2, 'w','LineStyle','none'); hold on;
violin_tool_mean_x_2 =  (violin_tool_mean_x-Xc)*cos(ang_tool) + (violin_tool_mean_y-Yc)*sin(ang_tool) + Xc;
violin_tool_mean_y_2 = -(violin_tool_mean_x-Xc)*sin(ang_tool) + (violin_tool_mean_y-Yc)*cos(ang_tool) + Yc;
violin_tool_mean_center = [mean(violin_tool_mean_x_2), mean(violin_tool_mean_y_2)];
% plot(violin_tool_mean_x_2, violin_tool_mean_y_2, 'k', 'LineWidth', 2.5); hold on;

ang_vehicle = 40.21;
violin_vehicle_x_2 =  (violin_vehicle_x-Xc)*cos(ang_vehicle) + (violin_vehicle_y-Yc)*sin(ang_vehicle) + Xc;
violin_vehicle_y_2 = -(violin_vehicle_x-Xc)*sin(ang_vehicle) + (violin_vehicle_y-Yc)*cos(ang_vehicle) + Yc;
fill(violin_vehicle_x_2, violin_vehicle_y_2, 'w','LineStyle','none'); hold on;
violin_vehicle_mean_x_2 =  (violin_vehicle_mean_x-Xc)*cos(ang_vehicle) + (violin_vehicle_mean_y-Yc)*sin(ang_vehicle) + Xc;
violin_vehicle_mean_y_2 = -(violin_vehicle_mean_x-Xc)*sin(ang_vehicle) + (violin_vehicle_mean_y-Yc)*cos(ang_vehicle) + Yc;
violin_vehicle_mean_center = [mean(violin_vehicle_mean_x_2), mean(violin_vehicle_mean_y_2)];
% plot(violin_vehicle_mean_x_2, violin_vehicle_mean_y_2, 'k', 'LineWidth', 2.5); hold on;

% Define colors
myColor = [ 166,206,227; 31,120,180; 178,223,138; 51,160,44; 251,154,153;
    227,26,28; 253,191,111; 255,127,0; 202,178,214; 106,61,154;
    255,255,153; 177,89,40; 141,211,199; 255,255,179; 190,186,218;
    251,128,114; 128,177,211; 253,180,98; 179,222,105; 252,205,229;
    217,217,217; 188,128,189; 204,235,197; 255,237,111; ]/255;

% Plot
for ca = 1:size(MCC_2_overall, 1)
    MCCpgon2 = polyshape(MCC_2_overall(ca, :)' .* xVector, MCC_2_overall(ca,:)' .* yVector);
    pg = plot(MCCpgon2, 'LineWidth', 1.2); hold on;
    pg.FaceAlpha = 0;
    %     pg.EdgeAlpha = 0.5;
    pg.EdgeColor = myColor(ca, :);

    plot(MCC_2_overall(ca, :)' .* xVector, MCC_2_overall(ca,:)' .* yVector, '.', 'color', myColor(ca, :))
    set(gca, 'box', 'on', 'xtick', [], 'ytick', [], 'fontSize', 24, 'fontName', 'Arial')
end