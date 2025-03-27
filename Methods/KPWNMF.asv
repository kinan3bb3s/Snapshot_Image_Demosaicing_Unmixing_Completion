% KPWNMF.m
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

function [I_WNMF_rec, G_WNMF, F_WNMF, F_pool, Final_norm, WNMF_toc] = KPWNMF(I_MOS_seq, SMP_seq, FilterPattern_Final, num_band, WNMF_params, num_obs_pxl, I_WB, smp_scenario)
    % KPWNMF: Performs demosaicing and unmixing using Weighted NMF with K-means.
    %
    % Inputs:
    %   I_MOS_seq: Mosaic image sequence
    %   SMP_seq: Sampling operators
    %   FilterPattern_Final: Filter pattern for each pixel
    %   num_band: Number of spectral bands
    %   WNMF_params: Parameters for WNMF
    %   num_obs_pxl: Number of observed pixels
    %   I_WB: Weighted bilinear interpolation result
    %   smp_scenario: Sampling scenario
    %
    % Outputs:
    %   I_WNMF_rec: Reconstructed image
    %   G_WNMF: Abundance matrix
    %   F_WNMF: Endmember matrix
    %   F_pool: Pool of endmembers
    %   Final_norm: Final norm value
    %   WNMF_toc: Execution time

    disp('Running WNMF with K-means');
    % Ensure K-means is enabled for KPWNMF
    WNMF_params.Kmeans = true;
    WNMF_params.global_rank = WNMF_params.global_rank; % Ensure global rank is set

    tic;
    % Perform demosaicing
    [I_WNMF_rec, F_pool, Final_norm] = WNMF_Demosaicing(I_MOS_seq, SMP_seq, FilterPattern_Final, num_band, WNMF_params, num_obs_pxl, I_WB, smp_scenario);
    % Perform unmixing
    [~, G_WNMF, F_WNMF] = low_rank_completion_methods(I_MOS_seq, SMP_seq, num_band, WNMF_params, num_obs_pxl, I_WNMF_rec, F_pool, Final_norm);
    WNMF_toc = toc;
end