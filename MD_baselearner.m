classdef MD_baselearner
    % This is the main class of the FIRST layer (base learner) of the 
    % double-layer multi-resolution memory decoding model
    % Author: Xiwei She
    
    properties
        % case information
        Category % category for decoding
        decodingCase % run case for decoding
        oF % output file
        
        % fitting options
        num_split % number of bagging split
        resolution_all % array with resolutions of b splines
        d % order of b splines
        L % memory length of the event in ms
        lambda_pool % lambda for lasso regularization
        par % this indicates if fitting is done in a parfor loop or not
        
        % intermediate variables
        target % this is the category classification used in estimation
        tFit % elapsed computing time
        R_first % First Layer Results

        % Input output variables
        TrainingSet_SpikeTensor
        TrainingSet_target
        TestingSet_SpikeTensor
        TestingSet_target
        CrossValSet
        
        % Case varialbes
        randomShuffle
        
        % Cluster variable
        jobName
    end
    
    methods
        %% File management
        function obj = MD_baselearner(Category, Split, runCase, varargin)
            [obj.num_split, obj.resolution_all, obj.d,obj.L, obj.par, obj.lambda_pool, obj.jobName] = process_options(varargin,...
                'num_split',5,'resolution_all',0:50,'d',3,'L',2000,'par',0, 'lambda_pool', power(10, 1:-0.1:-9), 'jobName', 'Job2');
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
            
            % Load Data
            iF1 = strcat('processedData\', thisCase,'\MD_', Category,'_split', mat2str(Split),'.mat');
            load(iF1, 'target', 'CrossValSet', 'TrainingSet_SpikeTensor', 'TrainingSet_target', 'TestingSet_SpikeTensor', 'TestingSet_target');
            
            
            %store object properties
            obj.target = target;
            obj.Category = Category;
            obj.decodingCase = thisCase;
            obj.TrainingSet_SpikeTensor = TrainingSet_SpikeTensor;
            obj.TrainingSet_target = TrainingSet_target;
            obj.TestingSet_SpikeTensor = TestingSet_SpikeTensor;
            obj.TestingSet_target = TestingSet_target;
            obj.CrossValSet = CrossValSet;
            
            %specify output file and save memory decoding setup
            obj.oF = strcat('result\MD_baselearner_',thisCase(3:end), '_', Category,'_fold',mat2str(Split), '.mat');
            MD_baselearner = obj;
            save(obj.oF,'MD_baselearner')
        end
        
        %% Get input signals
        function SpikeTensor = getSpikeTensor(obj)
            SpikeTensor = obj.TrainingSet_SpikeTensor;
        end
        
        %% A single split modeling
        function [firstR] = runSplit(obj,ti,SpikeTensor,varargin)
            % Training output labels
            thisSplit_c = obj.TrainingSet_target;
            
            % First Layer MD training
            for mi = 1:length(obj.resolution_all)
                m = obj.resolution_all(mi);
                P = SpikeTensor2BSplineFeatureMatrix(SpikeTensor, m, obj.d);
                
                % Random Shuffle Control only
                if obj.randomShuffle == 1
                    shuffleSeed = randperm(size(thisSplit_c, 1));
                    thisSplit_c = thisSplit_c(shuffleSeed);
                end
                
                % Randomize CV for this bagging split
                rng('shuffle');
                
                % The first layer modeling
                [B, FitInfo] = lassoglm(P, thisSplit_c, 'binomial','CV', 5, 'Lambda', obj.lambda_pool);
                firstR{mi}.Resolution = mi; % the b-spline resolution
                firstR{mi}.B = B;
                firstR{mi}.FitInfo = FitInfo;
                firstR{mi}.feature = P;
                firstR{mi}.target = thisSplit_c;
                
                % Show Results - First Layer
                fprintf('Done Split:%d; Resolution:%d;\n',ti, m);
                
            end
            
        end
        
        %% Run all splits
        function firstR = runAllSplits(obj,varargin)
            SpikeTensor=obj.getSpikeTensor;
            R_sepTrial = cell(obj.num_split, 1);
            if obj.par
                parfor ti = 1:obj.num_split
                    R_sepTrial{ti} = obj.runSplit(ti,SpikeTensor);
                end
            else
                for ti = 1:obj.num_split
                    R_sepTrial{ti} = obj.runSplit(ti,SpikeTensor);
                end
            end
            %Put R back in the original format
            for ti = 1:obj.num_split
                % prepare spatio-temporal patterns for classification
                for mi = 1:length(obj.resolution_all)
                    firstR(ti,mi) = R_sepTrial{ti}(mi);
                end
            end
            
        end
        
        %% Run the entire MD modeling process in parallel
        function MD_baselearner = run(obj,varargin)
            if obj.par
                poolOb = obj.setupPool;
                pctRunOnAll warning off
            end
            tStart = tic;
            obj.R_first = obj.runAllSplits();
            
            obj.tFit = toc(tStart);
            MD_baselearner = obj;
            save(obj.oF,'MD_baselearner', '-v7.3')
            if obj.par
                poolOb.delete;
            end
        end
        
        %% Parallel Computing
        % Here, the default is to put a seperate node per trial, with the number of workers equal to the number of bagging splits
        % Can run it on a cluster like the USC CARC or local PC with multiple CPU
        function poolOb = setupPool(obj)
            if ~isunix
                poolOb = parpool;
            else  
                % ------------------------ New for USC Slurm ------------------
                cluster = parallel.cluster.Slurm;
                
                job_folder = fullfile('/project/berger_92/xiweishe/SubmittedJob', getenv('SLURM_JOB_ID'));
                mkdir(job_folder);
                % set(cluster, 'Name', jobName)
                set(cluster, 'JobStorageLocation', job_folder);
                set(cluster, 'HasSharedFilesystem', true);
                job_argument = fullfile('--time=10:00:00 --mem-per-cpu=4GB --job-name=', obj.jobName);
                set(cluster, 'SubmitArguments', job_argument);
                % set(cluster, 'ResourceTemplate', '--ntasks=^nWorkers^');
                % clusProf = get_SLURM_cluster('/home/rcf-proj/tb/xiweishe/matlab_storage','/usr/usc/matlab/R2018a/SlurmIntegrationScripts','--time=50:00:00 --partition berger');
                % ------------------------ New for Slurm ------------------
            end
        end
    end
    
end

