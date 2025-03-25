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
function [I_WNMF_rec,F_pool,Final_norm] = WNMF_Demosaicing(I_MOS_seq,SMP_seq,FilterPattern_Cell,num_band,WNMF_params,num_obs_pxl,I_WB,smp_scenario)

disp('Initializing Values');
[n1,n2,n3]=size(I_MOS_seq);

offset=WNMF_params.WNMF_Offset;
r = WNMF_params.rank;
m = offset*offset; n = num_band; % size of the data matrix X (m x n)
Scaling=WNMF_params.Scaling;
I_WB_Init=WNMF_params.I_WB_Initialization;
% if(~isempty(WNMF_params.SMP_seq_Ideal))
%     SMP_seq_Ideal=WNMF_params.SMP_seq_Ideal;
% end

% Initializing G and F
Ginit = rand(m,r)*10+0.1;
Ginit=ScaleRows(Ginit);
Finit = rand(r,n)+0.1;

% Dividing the images into patches
step_size=WNMF_params.Step_size;
block_iter=1;
for xx=1:step_size:size(I_MOS_seq,1)-offset+1
    for yy=1:step_size:size(I_MOS_seq,2)-offset+1
        %tmp2  size is 5*5*2,
        tmp2=I_MOS_seq(xx:xx+offset-1,yy:yy+offset-1,:);
        %CurrBlock_lst is a cell array, the cell size is 25*2
        CurrBlock_lst{block_iter}=reshape(tmp2,[offset*offset,num_obs_pxl]);
        %Prepare filter pattern for WNMF
        tmpf=FilterPattern_Cell(xx:xx+offset-1,yy:yy+offset-1,:);
        CurrBlock_Filter_1st{block_iter}=reshape(tmpf,[offset*offset,num_obs_pxl]);
        
        tmp2b=I_WB(xx:xx+offset-1,yy:yy+offset-1,:);
        INT{block_iter}=reshape(tmp2b,[offset*offset,num_band]);
        %tmp3 is the sampling matrix , size is 5*5*16*2,
        tmp3=SMP_seq(xx:xx+offset-1,yy:yy+offset-1,:,:);
        %tmp4 size is 25*16*2
        tmp4=reshape(tmp3,[offset*offset,num_band,size(tmp3,4)]);
        FilterPattern_lst{block_iter}=tmp4;
        Loc{block_iter}=[xx,yy];
        
%         tmp5=SMP_seq_Ideal(xx:xx+offset-1,yy:yy+offset-1,:,:);
%         %tmp4 size is 25*16*2
%         tmp6=reshape(tmp5,[offset*offset,num_band,size(tmp5,4)]);
%         FilterPattern_lst_Ideal{block_iter}=tmp6;
        
        block_iter=block_iter+1;
    end
end



%% Demosaicing Stage 
disp('Demosaicing');
I_rec0=zeros(n1,n2,num_band);
D_rec0=cell(numel(FilterPattern_lst),1);% To store the completed patches
Final_G=[];
F_pool=[];
Final_norm=[];
G_New=[];
F_New=[];
parfor tt=1:numel(FilterPattern_lst)
    %     fprintf('\n ### Running WNMF , iteration number %d',tt);
    M2=CurrBlock_lst{tt}; % The patch 
    WB=INT{tt}; % The Weighted Bilinear interpolation output for the initialization   
    smp2=FilterPattern_lst{tt}; % The sampling matrix
%     smp3=FilterPattern_lst_Ideal{tt};
    if(smp_scenario==1)
        FinalMatrix=applyBWR(smp2,M2);% Expanding and Unfolding the patch
    else
        FinalMatrix=WB;% Taking the values from interpolation
%         D=eye(num_band,num_band);
%             for i=1:(num_band-1)
%                 for j=1:num_band
%                     if(i==j)
%                         D(i,j+1)=-1;
%                     end
%                 end
%             end
%             smp3=[smp2;0.0005*D];
%             e=zeros(num_band,1);
%         for gg=1:size(FinalMatrix,1)
%             ty=FinalMatrix(gg,:)';
%             ty=[ty;e];
%             qq=lsqnonneg(smp3,ty);
%             FinalMatrix2(gg,:)=qq';
%             
%         end
    end
    
    %Sampling matrix has to deal with multiple exposures.
    % Update the weight matrix in case of multiple exposures
    smp_temp=zeros(m,n,num_obs_pxl);
    for ii=1:num_obs_pxl
        smp_temp=smp_temp(:,:,1)+smp2(:,:,ii);
    end
    smp_final=squeeze(smp_temp(:,:,1));
    
    if(I_WB_Init) % If we need to initialize using Weighted Bilinear Interpolaztion
        % Unmix the rank-1 WB patch 
        [Ginit , Finit ]=NeNMF(WB,r,'MAX_ITER',10000,'MIN_ITER',10,'TOL',1e-1);
%         [~, K] = VCA(WB','Endmembers',r,'verbose','off');
%         Finit=WB(K,:);
%         I_WB=reshape(WB,[offset,offset,num_band]);
% %         Finit=ppi(I_WB,r)';
%         Ginit=estimateAbundanceLS(I_WB,Finit','Method','ncls');
%         Ginit=Ginit(:);
        if(sort(Finit)==Finit) % If all the values in Finit are equal, initialize from patch
            Finit=sum(FinalMatrix);
        end
        if(sort(round(Ginit,12))==round(Ginit,12))% If all the values in Ginit are equal, initialize randomly
            Ginit=Ginit+rand(num_band,1)/100;            
        end
        
        G_New=Finit';% G_New now is the endmemebers
        F_New=Ginit';% F_New now is the abundances
        [D_rec0{tt},G_New,F_New,t , initf , f] = WNMFLibrary.EM_WNE_NMF( smp_final' , FinalMatrix' , G_New,F_New, WNMF_params.Iter_max_M_step , WNMF_params.Iter_max_E_step , WNMF_params.Nesterov_Max_Iterations , WNMF_params.Nesterov_Min_Iterations,Scaling);
        [D_rec0{tt}]=D_rec0{tt}';
        F_pool=[F_pool;G_New']; % Store the spectrum for every patch
        Final_norm=[Final_norm;f];% Store the approximation error for every patch
    else % if we need to initialize randomly
        fprintf(' ### Random initialization for G and F ...');
        Finit=sum(FinalMatrix);
        G_New=Finit';
        F_New=Ginit';
        [D_rec0{tt},G_New,F_New,t , initf , f] = WNMFLibrary.EM_WNE_NMF( smp_final' , FinalMatrix' , F_New',G_New', WNMF_params.Iter_max_M_step , WNMF_params.Iter_max_E_step , WNMF_params.Nesterov_Max_Iterations , WNMF_params.Nesterov_Min_Iterations,Scaling);        
        [D_rec0{tt}]=D_rec0{tt}';
        F_pool=[F_pool;G_New'];
        Final_norm=[Final_norm;f];
    end
    
end
% Aggregate the patches together in 3d datacube
CNT=zeros(n1,n2,num_band);
for tt=1:numel(FilterPattern_lst)
    tmp=reshape(D_rec0{tt},[offset,offset,num_band]);
    
    I_rec0(Loc{tt}(1):Loc{tt}(1)+offset-1,Loc{tt}(2):Loc{tt}(2)+offset-1,:)=tmp+I_rec0(Loc{tt}(1):Loc{tt}(1)+offset-1,Loc{tt}(2):Loc{tt}(2)+offset-1,:);
    CNT(Loc{tt}(1):Loc{tt}(1)+offset-1,Loc{tt}(2):Loc{tt}(2)+offset-1,:)=CNT(Loc{tt}(1):Loc{tt}(1)+offset-1,Loc{tt}(2):Loc{tt}(2)+offset-1,:)+ones(offset,offset,num_band);
end
I_WNMF_rec=I_rec0./CNT;

% I_WNMF_rec(1:step_size,:,:)=I_WB(1:step_size,:,:);
% I_WNMF_rec(:,1:step_size,:)=I_WB(:,1:step_size,:);
% I_WNMF_rec(end-step_size-1:end,:,:)=I_WB(end-step_size-1:end,:,:);
% I_WNMF_rec(:,end-step_size-1:end,:)=I_WB(:,end-step_size-1:end,:);

end

