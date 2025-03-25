% LOCALLY-RANK-ONE-BASED JOINT UNMIXING AND DEMOSAICING METHODS FOR SNAPSHOT SPECTRAL IMAGES
% Implementation inspired by:
% Abbas, K., Puigt, M., Delmaire, G., & Roussel, G. (2024). 
% "Locally-Rank-One-Based Joint Unmixing and Demosaicing Methods for Snapshot Spectral Images. Part II: A Filtering-Based Framework"
% IEEE Transactions on Computational Imaging, 10, 806-817. DOI: 10.1109/TCI.2024.3402441
%
% Copyright (c) Kinan Abbas 2024
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
function [I_WNMF_rec, G, F] = Naive_WNMF(I_MOS_seq, SMP_seq, FilterPattern_Cell, num_band, WNMF_params, num_obs_pxl, I_WB, smp_scenario)
    % Naive Weighted Non-negative Matrix Factorization
    % This code divide the image into patches depending on the step size.
    % To process the full image, make step_size equal to the image
    % dimension
    disp('Initializing Values');
    
    % Get dimensions of the input mosaic image sequence
    [n1, n2, ~] = size(I_MOS_seq);
    
    % Extract parameters from WNMF_params structure
    offset = WNMF_params.WNMF_Offset;               % Size of patch (offset x offset)
    m = offset * offset;                            % Number of pixels in a patch
    n = num_band;                                   % Number of spectral bands
    Scaling = WNMF_params.Scaling;                  % Scaling flag for WNMF
    I_WB_Init = WNMF_params.I_WB_Initialization;    % Flag for WB initialization
    Ginit = WNMF_params.Ginit;                      % Initial abundance matrix
    Finit = WNMF_params.Finit;                      % Initial endmember matrix
    step_size = WNMF_params.Step_size;              % Step size for patch sliding
    
    % Handle special sampling scenario (smp_scenario == 3)
    if smp_scenario == 3
        disp('Processing special sampling scenario');
        if ~isempty(WNMF_params.SMP_seq_Ideal)
            SMP_seq_Ideal = WNMF_params.SMP_seq_Ideal; % Ideal sampling sequence if provided
        end
    end
    
    % Divide image into patches for processing
    % This section creates overlapping patches of size offset x offset
    block_iter = 1;  % Counter for number of patches
    for xx = 1:step_size:(size(I_MOS_seq,1)-offset+1)
        for yy = 1:step_size:(size(I_MOS_seq,2)-offset+1)
            % Extract patch from mosaic image sequence
            tmp2 = I_MOS_seq(xx:xx+offset-1, yy:yy+offset-1, :);
            CurrBlock_lst{block_iter} = reshape(tmp2, [m, num_obs_pxl]);
            
            % Extract corresponding filter pattern for the patch
            tmpf = FilterPattern_Cell(xx:xx+offset-1, yy:yy+offset-1, :);
            CurrBlock_Filter_1st{block_iter} = reshape(tmpf, [m, num_obs_pxl]);
            
            % Extract patch from Weighted Bilinear interpolated image
            tmp2b = I_WB(xx:xx+offset-1, yy:yy+offset-1, :);
            INT{block_iter} = reshape(tmp2b, [m, num_band]);
            
            % Process sampling matrix for the patch
            tmp3 = SMP_seq(xx:xx+offset-1, yy:yy+offset-1, :, :);
            tmp4 = reshape(tmp3, [m, num_band, size(tmp3,4)]);
            FilterPattern_lst{block_iter} = tmp4;
            Loc{block_iter} = [xx, yy];  % Store patch location
            
            % Handle ideal sampling matrix for scenario 3
            if smp_scenario == 3
                tmp5 = SMP_seq_Ideal(xx:xx+offset-1, yy:yy+offset-1, :, :);
                tmp6 = reshape(tmp5, [m, num_band, size(tmp5,4)]);
                FilterPattern_lst_Ideal{block_iter} = tmp6;
            end
            
            block_iter = block_iter + 1;
        end
    end
    
    %% Demosaicing Stage
    disp('Demosaicing');
    I_rec0 = zeros(n1, n2, num_band);  % Initialize reconstructed image
    D_rec0 = cell(numel(FilterPattern_lst), 1);  % Store completed patches
    
    % Process each patch individually
    for tt = 1:numel(FilterPattern_lst)
        M2 = CurrBlock_lst{tt};      % Current patch from mosaic image
        WB = INT{tt};                % WB interpolation output for initialization
        smp2 = FilterPattern_lst{tt}; % Sampling matrix for the patch
        smp2 = smp2.^3;              % Apply cubic weighting to sampling matrix
        
        % Handle ideal sampling for scenario 3
        if smp_scenario == 3
            smp3 = FilterPattern_lst_Ideal{tt};
        end
        
        % Prepare final matrix based on sampling scenario
        if smp_scenario == 1
            FinalMatrix = applyBWR(smp2, M2);  % Use BWR method for scenario 1
        else
            FinalMatrix = WB;                  % Use WB interpolation otherwise
        end
        
        % Combine multiple exposures in sampling matrix
        smp_temp = zeros(m, n, num_obs_pxl);
        for ii = 1:num_obs_pxl
            smp_temp(:,:,1) = smp_temp(:,:,1) + smp2(:,:,ii);
        end
        smp_final = squeeze(smp_temp(:,:,1));  % Final sampling matrix
        
        % Perform WNMF on the patch
        if I_WB_Init
            % Initialize using Weighted Bilinear Interpolation
            [Ginit, Finit] = NeNMF(WB, WNMF_params.global_rank, ...
                'MAX_ITER', 10000, ...         % Maximum iterations
                'MIN_ITER', 10, ...            % Minimum iterations
                'TOL', 1e-1);                  % Tolerance
            
            G_New = Finit';                    % Endmembers
            F_New = Ginit';                    % Abundances
            [D_rec0{tt}, G_New, F_New, ~, ~, ~] = WNMFLibrary.EM_WNE_NMF(...
                smp_final', FinalMatrix', G_New, F_New, ...
                WNMF_params.Iter_max_M_step, WNMF_params.Iter_max_E_step, ...
                WNMF_params.Nesterov_Max_Iterations, WNMF_params.Nesterov_Min_Iterations, ...
                Scaling);
            D_rec0{tt} = D_rec0{tt}';         % Transpose result
        else
            % Random initialization
            fprintf(' ### Random initialization for G and F ...');
            Finit = sum(FinalMatrix);          % Initialize F from patch sum
            G_New = Finit';
            F_New = Ginit';
            [D_rec0{tt}, G_New, F_New, ~, ~, ~] = WNMFLibrary.EM_WNE_NMF(...
                smp_final', FinalMatrix', F_New', G_New', ...
                WNMF_params.Iter_max_M_step, WNMF_params.Iter_max_E_step, ...
                WNMF_params.Nesterov_Max_Iterations, WNMF_params.Nesterov_Min_Iterations, ...
                Scaling);
            D_rec0{tt} = D_rec0{tt}';         % Transpose result
        end
    end
    
    %% Store results
    G = F_New';      % Final abundance matrix
    F = G_New';      % Final endmember matrix
    
    % Aggregate patches into final image
    CNT = zeros(n1, n2, num_band);  % Count array for averaging overlapping regions
    for tt = 1:numel(FilterPattern_lst)
        tmp = reshape(D_rec0{tt}, [offset, offset, num_band]);
        % Add patch to reconstructed image
        I_rec0(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :) = ...
            tmp + I_rec0(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :);
        % Update count for averaging
        CNT(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :) = ...
            CNT(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :) + ones(offset, offset, num_band);
    end
    
    % Normalize by count to handle overlapping regions
    I_WNMF_rec = I_rec0 ./ CNT;
end