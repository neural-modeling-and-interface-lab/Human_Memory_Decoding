classdef MD_metalearner
    % This is the main class of the SECOND layer of the 
    % double-layer multi-resolution memory decoding model
    % Author: Xiwei She
    
    properties
        % case information
        Category % category for decoding
        decodingCase % run case for decoding
        oF % output file
        
        % fitting options
        num_split % number of bagging split
        lambda_pool % lambda for lasso regularization
        par % this indicates if fitting is done in a parfor loop or not
        
        % intermediate variables
        tFit % elapsed computing time
        R_second % Second Layer Results

        % Input output variables
        TrainingSet_target
        TestingSet_target
        yProb_training
        yProb_testing
        B_firstLayer_global
        C0_firstLayer_global
        
        % Case varialbes
        randomShuffle
    end
    
    methods
        %% File management
        function obj = MD_metalearner(Category, Split, runCase, varargin)
            [obj.num_split, obj.par, obj.lambda_pool] = process_options(varargin, ...
                'num_split',5, 'par',0, 'lambda_pool', power(10, 1:-0.1:-6));
            warning off;
            
            % Determine decoding information
            switch (runCase)
                case 1
                    thisCase = '1 Sample Response';
                    obj.randomShuffle = 0;
                case 2
                    thisCase = '2 Match Response';
                    obj.randomShuffle = 0;
                case 3
                    thisCase = '3 Shifted Control';
                    obj.randomShuffle = 0;
                case 4
                    thisCase = '4 Shuffle Control';
                    obj.randomShuffle = 1;
                otherwise
                    thisCase = 'UNDEFINED CASE!';
            end
            
            % Load Input Data
            iF1 = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_fold',mat2str(Split), '_Parameters.mat');
            load(iF1, 'yProb_training', 'yProb_testing', 'B_global', 'C0_global');
            
            % Load Output Data
            iF2 = strcat('processedData\', thisCase,'\MD_', Category,'_split', mat2str(Split),'.mat');
            load(iF2, 'TrainingSet_target', 'TestingSet_target');
            
            %store object properties
            obj.Category = Category;
            obj.decodingCase = thisCase;
            obj.B_firstLayer_global = B_global;
            obj.C0_firstLayer_global = C0_global;
            obj.TestingSet_target = TestingSet_target;
            obj.TrainingSet_target = TrainingSet_target;
            
            % Get the second layer features
            yProb_training_temp = [];
            yProb_testing_temp = [];
            for resolution = 1:size(B_global, 1)
                yProb_training_temp = [yProb_training_temp, yProb_training{resolution, Split}];
                yProb_testing_temp = [yProb_testing_temp, yProb_testing{resolution, Split}];
            end
            obj.yProb_training = yProb_training_temp;
            obj.yProb_testing = yProb_testing_temp;
            
            %specify output file and save memory decoding setup
            obj.oF = strcat('result\MD_metalearner_',thisCase(3:end), '_', Category,'_fold',mat2str(Split), '.mat');
            MD_metalearner = obj;
            save(obj.oF,'MD_metalearner')
        end
        
        %% A single split modeling
        function [secondR] = runSplit(obj,ti,varargin)

            % Training output labels
            thisSplit_c = obj.TrainingSet_target;
            
            % Second Layer MD training
            thisSplit_feature = obj.yProb_training;
            
            % Random Shuffle Control only
            if obj.randomShuffle == 1
                shuffleSeed = randperm(size(thisSplit_c, 1));
                thisSplit_c = thisSplit_c(shuffleSeed);
            end
                
            % Randomize CV for this bagging split
            rng('shuffle');
            
            % The second layer modeling
            [B_second, FitInfo_second] = lassoglm(thisSplit_feature, thisSplit_c, 'binomial','CV', 5, 'Lambda', obj.lambda_pool);
            
            % Save Results
            secondR.B_second = B_second;
            secondR.FitInfo_second = FitInfo_second;
            secondR.feature = thisSplit_feature;
            secondR.target = thisSplit_c;
            
            % Show Results
            fprintf('Done Split:%d; \n',ti);
            
        end
        
        %% Run all splits
        function secondR = runAllSplits(obj,varargin)
            R_sepTrial = cell(obj.num_split, 1);
            if obj.par
                spmd
                    warning('off')
                end
                parfor ti = 1:obj.num_split
                    R_sepTrial{ti} = obj.runSplit(ti);
                end
            else
                for ti = 1:obj.num_split
                    R_sepTrial{ti} = obj.runSplit(ti);
                end
            end
            %Put R back in the original format
            for ti = 1:obj.num_split
                secondR{ti} = R_sepTrial{ti};
            end
            
        end
        
        %% Run the entire MDM modeling process
        function MD_metalearner = run(obj,varargin)
            if obj.par
                poolOb = obj.setupPool;
            end
            tStart = tic;
            obj.R_second = obj.runAllSplits();
            
            obj.tFit = toc(tStart);
            MD_metalearner = obj;
            save(obj.oF,'MD_metalearner', '-v7.3')
            if obj.par
                poolOb.delete;
            end
        end
        
        %% On CARC Parallel Computing
        % here, the default is to put a seperate node per trial, with the number of workers equal to the number of bagging splits
        % Can run it on a cluster like the USC CARC or local PC with multiple CPU
        function poolOb = setupPool(obj)
            if ~isunix
                poolOb = parpool;
            else  
                nWorkers = obj.num_split;
                % ------------------------ New for Slurm ------------------
                clusProf = get_SLURM_cluster('/home/rcf-proj/tb/xiweishe/matlab_storage','/usr/usc/matlab/R2018a/SlurmIntegrationScripts','--time=50:00:00 --partition berger');
                % ------------------------ New for Slurm ------------------
                poolOb = parpool(clusProf,nWorkers);
            end
        end
    end
    
end

