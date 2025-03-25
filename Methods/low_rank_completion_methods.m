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
function [Unmixing_Image, Final_G, Final_F] = low_rank_completion_methods(I_MOS_seq, SMP_seq, num_band, WNMF_params, num_obs_pxl, I_WNMF_rec, F_pool, Final_norm)
    % Input:
    % I_MOS_seq: The SSI image
    % SMP_seq: The sampling matrix
    % num_band: The number of wavelengths
    % WNMF_params: parameters required to run the algorithm:
    %   WNMF_params.Kmeans: true to apply Kmeans, false to apply VCA
    %   WNMF_params.Kmeans_cut: to determine the threshold for kmeans/VCA
    %       possible values: null for no cut, 1 for cut on the mean, 
    %       2 for cut on the sqrt of the mean, 3 half of the mean, 
    %       4 median, 5 the first 100 element
    %   WNMF_params.global_rank: the rank of the full image for clustering
    % num_obs_pxl: Number of scans of the scene
    % I_WNMF_rec: The completed images using WNMF method (3d datacube)
    % F_pool: The pool of spectra collected from the rank-1 patches
    % Final_norm: The approximation error for every spectrum in the F_pool
    %
    % Output:
    % Final_G: The abundances matrix
    % Final_F: The endmembers matrix
    % Unmixing_Image: The estimated image after multiplying Final_F*Final_G
    %
    % Author: Kinan ABBAS
    % Creation Date: OCT 5 2022
    % Update Date: OCT 10 2022
    % New update: 7 Jan 2023 - Use WNMF to calculate G with fixed F
    % New update: 30 Jan 2023 - Update Kmeans to use Kmeans++ as initialization method,
    % increased iterations, runs 10 times and selects lowest within-cluster sums

    [n1, n2, n3] = size(I_MOS_seq);
    
    %% Filtering the pool of the spectra
    F_pool_filtered = [];
    if (isempty(WNMF_params.Kmeans_cut) || WNMF_params.Kmeans_cut == 0)
        mean_F_pool = 500000000000000000000;
    elseif WNMF_params.Kmeans_cut == 1
        mean_F_pool = mean(Final_norm);
    elseif WNMF_params.Kmeans_cut == 2
        mean_F_pool = mean(Final_norm);
        if (mean_F_pool < 1)
            mean_F_pool = mean_F_pool * mean_F_pool;
        else
            mean_F_pool = sqrt(mean_F_pool);
        end
    elseif WNMF_params.Kmeans_cut == 3
        mean_F_pool = mean(Final_norm);
        if (mean_F_pool < 1)
            mean_F_pool = mean_F_pool * 2;
        else
            mean_F_pool = mean_F_pool / 2;
        end
    elseif WNMF_params.Kmeans_cut == 4
        mean_F_pool = median(Final_norm);
    elseif WNMF_params.Kmeans_cut == 5
        Final_norm_temp = sort(Final_norm);
        mean_F_pool = Final_norm_temp(WNMF_params.threshold);
    end
    
    j = 1;
    for i = 1:size(Final_norm,1)
        if (Final_norm(i) < mean_F_pool)
            F_pool_filtered(j,:) = F_pool(i,:);
            j = j + 1;
        end
    end
    
    %% Run the clustering stage to estimate the final endmembers Final_F variable
    disp('Clustering');
    % KPWNMF
    if (WNMF_params.Kmeans == true || isempty(WNMF_params.Kmeans)) % For running K-means
        disp('KPWNMF');
        [~, C] = kmeans(F_pool_filtered, WNMF_params.global_rank, ...
            'Distance', 'cityblock', ...
            'Options', statset('UseParallel', 1), ...
            'start', 'plus', ...
            'Replicates', 10, ...
            'MaxIter', 10000, ...
            'Display', 'final', ...
            'OnlinePhase', 'off');
        Final_F = C;
    else % VCA_PWNMF
        if (WNMF_params.pure_pixel_algorithm == 1)
            disp('VCA_PWNMF');
            [~, K] = VCA(F_pool_filtered', 'Endmembers', WNMF_params.global_rank, 'verbose', 'off');
        elseif (WNMF_params.pure_pixel_algorithm == 2)
            disp('Post-Prec-SPA');
            K = PostPrecSPA(F_pool_filtered', WNMF_params.global_rank, 0, 1, 0);
        elseif (WNMF_params.pure_pixel_algorithm == 3)
            disp('XRAY');
            kf = FastConicalHull(F_pool_filtered', WNMF_params.global_rank);
            K = kf';
        elseif (WNMF_params.pure_pixel_algorithm == 4)
            disp('RSPA');
            options.normalize = 0;
            K = RSPA(F_pool_filtered', WNMF_params.global_rank, options);
        elseif (WNMF_params.pure_pixel_algorithm == 5)
            disp('SPA');
            K = FastSepNMF(F_pool_filtered', WNMF_params.global_rank);
        elseif (WNMF_params.pure_pixel_algorithm == 6)
            disp('Post SPA');
            K = FastSepNMFpostpro(F_pool_filtered', WNMF_params.global_rank);
        elseif (WNMF_params.pure_pixel_algorithm == 7)
            disp('Prec SPA');
            K = PrecSPA(F_pool_filtered', WNMF_params.global_rank);
        end
        Final_F = F_pool_filtered(K,:);
    end
    
    %% Estimating the abundances
    disp('Estimating G');
    % Unfolding the 3d datacube
    A = reshape(I_WNMF_rec, [n1*n2, num_band]);
    
    % Unfolding the sampling matrix
    B = reshape(SMP_seq, [n1*n2, num_band, size(SMP_seq,4)]);
    
    % Process the multiple scans case for the sampling matrix
    smp_temp = zeros(n1*n2, n3, num_obs_pxl);
    for ii = 1:num_obs_pxl
        smp_temp = smp_temp(:,:,1) + B(:,:,ii);
    end
    B = squeeze(smp_temp(:,:,1));
    
    % Estimating G using NMF
    % Sum to one constraint
    A1 = [A, WNMF_params.alpha*ones(size(A,1),1)]; 
    Final_F1 = [Final_F, WNMF_params.gamma*ones(WNMF_params.global_rank,1)];
    
    [~, Final_G, iter, elapse, HIS] = NeNMF_Fixed_W(A1', WNMF_params.global_rank, ...
        'MAX_ITER', 1000, ...
        'MIN_ITER', 10, ...
        'W_FFIXID', Final_F1', ...
        'H_INIT', WNMF_params.Ginit', ...
        'SCALLING', false);
    Final_G = Final_G';
    
    %% Final Image
    WNMF_params1 = WNMF_params;
    WNMF_params1.WNMF_Offset = 100;
    WNMF_params1.rank = WNMF_params.global_rank;
    WNMF_params1.Iter_max_E_step = 100;
    WNMF_params1.Iter_max_M_step = 1;
    WNMF_params1.Nesterov_Max_Iterations = 1000;
    WNMF_params1.Nesterov_Min_Iterations = 10;
    WNMF_params1.Step_size = 100;
    WNMF_params1.Scaling = false;
    FilterPattern_Final = I_MOS_seq;
    WNMF_params1.Ginit = Final_G;
    WNMF_params1.Finit = Final_F;
    WNMF_params1.alpha = 20;
    WNMF_params1.gamma = 30;
    WNMF_params1.beta = 0;
    
    [I_WNMF_rec, Final_G, ~] = WNMF_Demosaicing_Round_2(I_MOS_seq, SMP_seq, ...
        FilterPattern_Final, num_band, WNMF_params1, num_obs_pxl, ...
        reshape(I_WNMF_rec, [size(I_MOS_seq,1), size(I_MOS_seq,2), num_band]), 1);
    
    for i = 1:size(Final_G,1)
        temp = Final_G(i,:);
        for j = 1:size(temp,2)
            if (temp(j) < 0.1)
                temp(j) = 0;
            end
        end
        Final_G(i,:) = temp;
    end
    
    Unmixing_Image = reshape(I_WNMF_rec, [size(I_MOS_seq,1)*size(I_MOS_seq,2), num_band]);
end