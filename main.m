% LOCALLY-RANK-ONE-BASED JOINT UNMIXING AND DEMOSAICING METHODS FOR SNAPSHOT SPECTRAL IMAGES
% Implementation inspired by:
% K. Abbas, M. Puigt, G. Delmaire, and G. Roussel (2024).
% "Locally-Rank-One-Based Joint Unmixing and Demosaicing Methods for Snapshot Spectral Images. Part I: A Matrix-Completion Framework."
% IEEE Transactions on Computational Imaging, 10, 848-862. DOI: 10.1109/TCI.2024.3402322
%
% MIT License
% Copyright (c) Kinan Abbas, Matthieu Puigt, Gilles Delmaire, and Gilles Roussel 2024
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

%% Cleaning and loading everything to the path
close all;
clear;
run_me_first;

%% Setting parameters for the experiment

% Parameters for KPWNMF
num_band = 25;                  % Number of spectral bands
sz = [100, 100];                % Image size
smp_scenario = 1;               % Sampling operator: 1 = binary (fixed for this run)
num_obs_pxl = 1;                % Number of exposures
WNMF_params.WNMF_Offset = sqrt(num_band);  % Patch dimensions
WNMF_params.rank = 1;           % Rank of the patch
WNMF_params.Iter_max_E_step = 25;  % Iterations for E-step in EM algorithm
WNMF_params.Iter_max_M_step = 1;   % Iterations for M-step in EM algorithm
WNMF_params.Nesterov_Max_Iterations = 1000;  % Max iterations for Nesterov
WNMF_params.Nesterov_Min_Iterations = 10;    % Min iterations for Nesterov
WNMF_params.beta = 0;           % Control for graph regularization and norms
WNMF_params.Step_size = WNMF_params.WNMF_Offset;  % No overlap between patches
WNMF_params.Scaling = false;    % No sum-to-one constraint on G
WNMF_params.I_WB_Initialization = true;  % Use WB output as initialization
unmixing = true;                % Enable unmixing after demosaicing
num_of_experiments = 1;         % Number of experiment repetitions
WNMF_params.Kmeans = true;      % Use K-means for clustering
WNMF_params.Kmeans_cut = 4;     % Median cut for K-means
WNMF_params.pure_pixel_algorithm = 1;  % VCA algorithm variant
WNMF_params.NesterovUnmixing = true;   % Use Nesterov for unmixing
WNMF_params.NesterovScaling = 15;      % Control sum-to-one constraint
WNMF_params.alpha = 20;         % Parameter for KPWNMF
WNMF_params.gamma = 30;         % Parameter for KPWNMF

Image_Selection = 2;            % 2: Abundances may or may not change assumption 

% Parameters for GRMR (kept for compatibility, though not used in visualization)
GRMR_params.offset = 5;
GRMR_params.maxIter = 20;
GRMR_params.sgm2 = 1e1;
GRMR_params.gamma = 0.2;
GRMR_params.rank_sel = 2;

%% Run the experiment
% Fixed to sampling scenario 1 without noise
Evaluate_on_complex_image(num_band, sz, smp_scenario, num_obs_pxl, GRMR_params, WNMF_params, unmixing, num_of_experiments, Image_Selection);