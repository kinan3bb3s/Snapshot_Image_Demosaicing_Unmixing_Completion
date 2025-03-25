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
function [I_WNMF_rec, G_Final, F_Final] = WNMF_Demosaicing_Round_2(I_MOS_seq, SMP_seq, FilterPattern_Cell, num_band, WNMF_params, num_obs_pxl, I_WB, smp_scenario)
    % Weighted Non-negative Matrix Factorization for image demosaicing (
    
    disp('Initializing Values');
    
    % Get dimensions of input mosaic image sequence
    [n1, n2, ~] = size(I_MOS_seq);
    
    % Extract parameters from WNMF_params structure
    offset = WNMF_params.WNMF_Offset;            % Patch size (offset x offset)
    r = WNMF_params.rank;                        % Rank for factorization
    m = offset * offset;                         % Number of pixels in a patch
    n = num_band;                                % Number of spectral bands
    Scaling = WNMF_params.Scaling;               % Scaling flag for WNMF
    I_WB_Init = WNMF_params.I_WB_Initialization; % Flag for WB initialization
    Ginit = WNMF_params.Ginit;                   % Initial abundance matrix
    Finit = WNMF_params.Finit;                   % Initial endmember matrix
    step_size = WNMF_params.Step_size;           % Step size for patch sliding
    
    % Divide image into patches
    block_iter = 1;
    for xx = 1:step_size:(size(I_MOS_seq,1)-offset+1)
        for yy = 1:step_size:(size(I_MOS_seq,2)-offset+1)
            % Extract patch from mosaic image
            tmp2 = I_MOS_seq(xx:xx+offset-1, yy:yy+offset-1, :);
            CurrBlock_lst{block_iter} = reshape(tmp2, [m, num_obs_pxl]);
            
            % Extract corresponding filter pattern
            tmpf = FilterPattern_Cell(xx:xx+offset-1, yy:yy+offset-1, :);
            CurrBlock_Filter_1st{block_iter} = reshape(tmpf, [m, num_obs_pxl]);
            
            % Extract patch from WB interpolated image
            tmp2b = I_WB(xx:xx+offset-1, yy:yy+offset-1, :);
            INT{block_iter} = reshape(tmp2b, [m, num_band]);
            
            % Process sampling matrix
            tmp3 = SMP_seq(xx:xx+offset-1, yy:yy+offset-1, :, :);
            tmp4 = reshape(tmp3, [m, num_band, size(tmp3,4)]);
            FilterPattern_lst{block_iter} = tmp4;
            Loc{block_iter} = [xx, yy];  % Store patch location
            
            block_iter = block_iter + 1;
        end
    end
    
    %% Demosaicing Stage
    disp('Demosaicing');
    
    % Initialize output arrays
    I_rec0 = zeros(n1, n2, num_band);           % Reconstructed image
    D_rec0 = cell(numel(FilterPattern_lst), 1); % Store completed patches
    F_pool = [];                                % Pool of endmember spectra
    Final_norm = [];                            % Approximation errors
    
    % Process each patch
    for tt = 1:numel(FilterPattern_lst)
        M2 = CurrBlock_lst{tt};      % Current patch
        WB = INT{tt};                % WB interpolation output
        smp2 = FilterPattern_lst{tt}; % Sampling matrix
        
        % Prepare final matrix based on sampling scenario
        if smp_scenario == 1
            FinalMatrix = applyBWR(smp2, M2);  % Use BWR method
        else
            FinalMatrix = WB;                  % Use WB interpolation
        end
        
        % Combine multiple exposures in sampling matrix
        smp_temp = zeros(m, n, num_obs_pxl);
        for ii = 1:num_obs_pxl
            smp_temp(:,:,1) = smp_temp(:,:,1) + smp2(:,:,ii);
        end
        smp_final = squeeze(smp_temp(:,:,1));
        
        % Perform WNMF
        if I_WB_Init
            % Initialize using Weighted Bilinear Interpolation
            G_New = Finit';  % Endmembers
            F_New = Ginit';  % Abundances
            [D_rec0{tt}, G_New, F_New, ~, ~, f] = WNMFLibrary.EM_WNE_NMF_Fixed_F(...
                smp_final', FinalMatrix', G_New, F_New, ...
                WNMF_params.Iter_max_M_step, WNMF_params.Iter_max_E_step, ...
                WNMF_params.Nesterov_Max_Iterations, WNMF_params.Nesterov_Min_Iterations, ...
                WNMF_params.alpha, WNMF_params.gamma, WNMF_params.beta);
            D_rec0{tt} = D_rec0{tt}';  % Transpose result
        else
            % Random initialization
            fprintf(' ### Random initialization for G and F ...');
            Finit = sum(FinalMatrix);
            G_New = Finit';
            F_New = Ginit';
            [D_rec0{tt}, G_New, F_New, ~, ~, f] = WNMFLibrary.EM_WNE_NMF(...
                smp_final', FinalMatrix', F_New', G_New', ...
                WNMF_params.Iter_max_M_step, WNMF_params.Iter_max_E_step, ...
                WNMF_params.Nesterov_Max_Iterations, WNMF_params.Nesterov_Min_Iterations, ...
                Scaling);
            D_rec0{tt} = D_rec0{tt}';  % Transpose result
        end
        
        % Store results for clustering
        F_pool = [F_pool; G_New'];
        Final_norm = [Final_norm; f];
    end
    
    % Store final matrices
    G_Final = F_New';  % Final abundance matrix
    F_Final = G_New';  % Final endmember matrix
    
    % Aggregate patches into final image
    CNT = zeros(n1, n2, num_band);  % Count array for averaging
    for tt = 1:numel(FilterPattern_lst)
        tmp = reshape(D_rec0{tt}, [offset, offset, num_band]);
        % Add patch to reconstructed image
        I_rec0(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :) = ...
            tmp + I_rec0(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :);
        % Update count for averaging
        CNT(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :) = ...
            CNT(Loc{tt}(1):Loc{tt}(1)+offset-1, Loc{tt}(2):Loc{tt}(2)+offset-1, :) + ones(offset, offset, num_band);
    end
    
    % Normalize overlapping regions
    I_WNMF_rec = I_rec0 ./ CNT;
end