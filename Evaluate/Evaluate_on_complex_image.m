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
% SOFTWARE..
function [] = Evaluate_on_complex_image(num_band, sz, smp_scenario, num_obs_pxl, GRMR_params, WNMF_params, unmixing, num_of_experiments, Image_Selection)

    % Test demosaicing and unmixing on complex Image, showing abundance maps and endmembers
    % Author: Kinan ABBAS
    % Creation Date: Oct 10 2022
    % Last modification date: Oct 10 2022
    
    show_Figuers = true;
    
    for zz = 1:num_of_experiments
        %% Loading the cube
        load('Water_Metal_Concret.mat')
        if (Image_Selection == 1)
            load('abundance_changing_image.mat');
        else
            load('abundance_complex_image.mat');
        end
        r = 3; % Fixed rank of the image
        Selected_Wavelengths = [12,50,93,147,180,201,263,304,317,329,395,464,569,600,632,650,662,665,700,710,718,783,799,840,845];
        
        if (num_band == 25)
            F_HS = F_HS(1:r, Selected_Wavelengths);
        else
            F_HS = F_HS(1:r, Selected_Wavelengths(3:18));
        end
        
        I_HS_Temp = G_HS * F_HS;
        I_HS = reshape(I_HS_Temp, [sz(1), sz(2), num_band]);
        
        %% Initializing some values
        [n1, n2, n3] = size(I_HS);
        m = n1 * n2; 
        n = num_band;
        Ginit = rand(m, r) * 10 + 0.1;
        Ginit = ScaleRows(Ginit);
        Finit = rand(r, n) + 0.0001;
        WNMF_params.Ginit = Ginit;
        WNMF_params.Finit = Finit;
        
        %% Applying the mosaic filter to acquire SSI image
        if num_band == 25
            load('spectral_responses_5x5.mat');
        elseif num_band == 16
            load('spectral_responses_4x4.mat');
            CentralWavelengths = CentralWavelength;
        else
            disp('Error');
        end
        
        temp2 = sort(round(CentralWavelengths)) - 400;
        SpectralProfiles = SpectralProfiles(:, temp2);
        SpectralProfiles = rot90(SpectralProfiles);
        
        [SMP_seq, FilterPattern_lst] = make_sampling_operators2(n1, n2, n3, num_obs_pxl, num_band, smp_scenario, SpectralProfiles);
        [I_MOS_seq] = acquire_observations(I_HS, SMP_seq, num_obs_pxl);
        
        %% Weighted bilinear interpolation (WB)
        disp('Running WB');
        tic;
        I_WB_tmp = zeros(sz(1), sz(2), num_band, num_obs_pxl);
        for pp = 1:num_obs_pxl
            I_MOS = I_MOS_seq(:,:,pp);
            FilterPattern = cell2mat(FilterPattern_lst(pp));
            I_WB_tmp(:,:,:,pp) = run_WB(I_MOS, FilterPattern, num_band);
        end
        I_WB = mean(I_WB_tmp, 4);
        if (unmixing)
            if (WNMF_params.NesterovUnmixing == true)
                [G_WB, F_WB, ~, ~, ~] = unmix(I_WB, r, Ginit, Finit, WNMF_params.NesterovScaling);
            else
                F_WB = ppi(I_WB, r)';
                abundanceMap = estimateAbundanceLS(I_WB, F_WB', 'Method', 'ncls');
                G_WB = reshape(abundanceMap, [n1*n2, r]);
            end
            WNMF_params.Ginit = G_WB;
        end
        WB_toc = toc;
        
        %% KPWNMF
        disp('Running WNMF with K-means');
        FilterPattern_Final = zeros(n1, n2, num_obs_pxl);
        WNMF_params.global_rank = r;
        for gg = 1:num_obs_pxl
            cc = cell2mat(FilterPattern_lst(gg));
            FilterPattern_Final(:,:,gg) = cc;
        end
        tic;
        [I_WNMF_rec, F_pool, Final_norm] = WNMF_Demosaicing(I_MOS_seq, SMP_seq, FilterPattern_Final, num_band, WNMF_params, num_obs_pxl, I_WB, smp_scenario);
        [~, G_WNMF, F_WNMF] = low_rank_completion_methods(I_MOS_seq, SMP_seq, num_band, WNMF_params, num_obs_pxl, I_WNMF_rec, F_pool, Final_norm);
        WNMF_toc = toc;
        
        %% VCA_PWNMF
        disp('Running WNMF with VCA');
        WNMF_params.Kmeans = false;
        WNMF_params.Kmeans_cut = 4;
        WNMF_params.threshold = 35;
        tic;
        [~, G_WNMF1, F_WNMF1] = low_rank_completion_methods(I_MOS_seq, SMP_seq, num_band, WNMF_params, num_obs_pxl, I_WNMF_rec, F_pool, Final_norm);
        WNMF_toc1 = toc;
        
        %% Naive WNMF
        disp('Running Naive WNMF');
        WNMF_params1 = WNMF_params;
        WNMF_params1.WNMF_Offset = sz(1);
        WNMF_params1.rank = r;
        WNMF_params1.Iter_max_E_step = 50;
        WNMF_params1.Iter_max_M_step = 1;
        WNMF_params1.Nesterov_Max_Iterations = 100;
        WNMF_params1.Nesterov_Min_Iterations = 10;
        WNMF_params1.global_rank = r;
        WNMF_params1.Kmeans = false;
        WNMF_params1.Scaling = true;
        tic;
        [~, G_WNMF2, F_WNMF2] = Naive_WNMF(I_MOS_seq, SMP_seq, FilterPattern_Final, num_band, WNMF_params1, num_obs_pxl, I_WB, smp_scenario);
        WNMF_toc2 = toc;
        
        %% Visualization
        if (show_Figuers)
            % Reshape abundance matrices for visualization
            G = reshape(G_HS, [sz(1), sz(2), r]);      % Ground truth abundance
            G_WB_reshaped = reshape(G_WB, [sz(1), sz(2), r]);  % WB abundance
            G_VCA = reshape(G_WNMF1, [sz(1), sz(2), r]);       % VCA_PWNMF abundance
            G_KPWNMF = reshape(G_WNMF, [sz(1), sz(2), r]);     % KPWNMF abundance
            G_Naive = reshape(G_WNMF2, [sz(1), sz(2), r]);     % Naive WNMF abundance
            
            % Abundance maps: Ground Truth vs KPWNMF
            figure('Name', 'Ground Truth vs KPWNMF Abundance Maps');
            subplot(3,2,1); imagesc(G(:,:,1)); title('GT Abundance 1');
            subplot(3,2,2); imagesc(G_KPWNMF(:,:,1)); title('KPWNMF Abundance 1');
            subplot(3,2,3); imagesc(G(:,:,2)); title('GT Abundance 2');
            subplot(3,2,4); imagesc(G_KPWNMF(:,:,2)); title('KPWNMF Abundance 2');
            subplot(3,2,5); imagesc(G(:,:,3)); title('GT Abundance 3');
            subplot(3,2,6); imagesc(G_KPWNMF(:,:,3)); title('KPWNMF Abundance 3');
            
            % Abundance maps: Ground Truth vs VCA_PWNMF
            figure('Name', 'Ground Truth vs VCA_PWNMF Abundance Maps');
            subplot(3,2,1); imagesc(G(:,:,1)); title('GT Abundance 1');
            subplot(3,2,2); imagesc(G_VCA(:,:,1)); title('VCA_PWNMF Abundance 1');
            subplot(3,2,3); imagesc(G(:,:,2)); title('GT Abundance 2');
            subplot(3,2,4); imagesc(G_VCA(:,:,2)); title('VCA_PWNMF Abundance 2');
            subplot(3,2,5); imagesc(G(:,:,3)); title('GT Abundance 3');
            subplot(3,2,6); imagesc(G_VCA(:,:,3)); title('VCA_PWNMF Abundance 3');
            
            % Abundance maps: Ground Truth vs WB
            figure('Name', 'Ground Truth vs WB Abundance Maps');
            subplot(3,2,1); imagesc(G(:,:,1)); title('GT Abundance 1');
            subplot(3,2,2); imagesc(G_WB_reshaped(:,:,1)); title('WB Abundance 1');
            subplot(3,2,3); imagesc(G(:,:,2)); title('GT Abundance 2');
            subplot(3,2,4); imagesc(G_WB_reshaped(:,:,2)); title('WB Abundance 2');
            subplot(3,2,5); imagesc(G(:,:,3)); title('GT Abundance 3');
            subplot(3,2,6); imagesc(G_WB_reshaped(:,:,3)); title('WB Abundance 3');
            
            % Abundance maps: Ground Truth vs Naive WNMF
            figure('Name', 'Ground Truth vs Naive WNMF Abundance Maps');
            subplot(3,2,1); imagesc(G(:,:,1)); title('GT Abundance 1');
            subplot(3,2,2); imagesc(G_Naive(:,:,1)); title('Naive WNMF Abundance 1');
            subplot(3,2,3); imagesc(G(:,:,2)); title('GT Abundance 2');
            subplot(3,2,4); imagesc(G_Naive(:,:,2)); title('Naive WNMF Abundance 2');
            subplot(3,2,5); imagesc(G(:,:,3)); title('GT Abundance 3');
            subplot(3,2,6); imagesc(G_Naive(:,:,3)); title('Naive WNMF Abundance 3');
            
            
        end
        
     
    end
end