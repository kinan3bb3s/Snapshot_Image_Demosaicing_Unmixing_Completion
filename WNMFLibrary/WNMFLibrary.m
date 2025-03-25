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
classdef WNMFLibrary
    %% This library contains different WNMF methods with multiple setups
    % 1) EM_WMU_NMF Expectation Maximization WNMF with multiplicative updates algorithm
    % 2) EM_WNE_NMF Expectation Maximization WNMF with Nesterov solver
    % 3) EM_L1R_WNE_NMF Expectation Maximization WNMF with Nesterov solver and
    % L1 norm
    % 5) EM_L2R_WNE_NMF Expectation Maximization WNMF with Nesterov solver and
    % L2 norm
    % 6) EM_GWNE_NMF Expectation Maximization WNMF with Nesterov solver and
    % Graph regularization
    % 7) EM_SPA_WNMF Expectation Maximization with SPA for estimating the
    % endmemebers and Nesterov with fixed F (the endmemebers) to estimate
    % the abundances.
    % 8) EM_WNE_NMF_Fixed_F EM_WNE_NMF Expectation Maximization WNMF with
    % Nesterov solver and with fixed F (endmembers)
    % 9) EM_PPI Expectation Maximization Expectation Maximization with PPI
    % (Purity Pixel Index) algorithm to estimate the endmemebes and Least
    % Squares to estimate the abundances.
    % 10) EM_WEucNMF Expectation Maximization with Entropy weighted non-negative matrix factorization

    % Created By Kinan ABBAS on July 2021
    % Last update date Oct 5 2022




    methods (Static)
        %%
        function [X , G , F , t , initf , f] = EM_WMU_NMF( W ,X , G , F ,Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations )
            initf = norm( W.*(X-G*F) , 'fro' )^2;
            W2 = W.^2;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            %E Step
            tic;
            for i=1:Iter_max_E_step
                WX = W2.*X;
                Xnew=WX+ (I-W).*(G*F);

                for j=1: Iter_max_M_step
                    %M Step
                    [ G , F ]=MU_NMF( Xnew , G , F , Iter_max_M_step );
                end
            end

            t = toc;
            f = norm( W.*(X-G*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );

        end
        %%
        function [Result , G, F , t , initf , f] = EM_WNE_NMF( W ,X , G , F ,  Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling )
            initf = norm( W.*(X-G*F) , 'fro' )^2;
            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                Result=WX+ (I-W).*(G*F);
                for j=1:Iter_max_M_step
                    %M Step
                    if(Scaling)
                        Result=[Result;20*ones(1,n2)]; G=[G;30*ones(1,reducedDim)];
                        [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'SCALLING',false,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                        Result=Result(1:end-1,:);
                        G=G(1:end-1,:);
                    else
                        [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                    end
                end
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            %             f = norm( W.*(Result-G*F) , 'fro' )^2;
            %             G_temp=ones(n1,reducedDim);
            %             f = norm( W.*(Result-G_temp*F) , 'fro' )^2;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            %             fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );
            %             num_band=size(Result,1);
            %             alpah=0.005;
            %             D=eye(num_band,num_band);
            %             for i=1:(num_band-1)
            %                 for j=1:num_band
            %                     if(i==j)
            %                         D(i,j+1)=-1;
            %                     end
            %                 end
            %             end
            %
            %             e=zeros(num_band,1);
            %             Z=W(:,1:num_band)';
            %             Z=[Z;alpah*D];
            %             for ii=1:size(Result,2)
            %
            %
            %                 Y=Result(ii,:)';
            %                 Y=[Y;e];
            %
            %                 Result(:,ii)=lsqnonneg(Z,Y);
            %             end

        end

        function [Result , G, F , t , initf , f] = EM_Abundances( W ,X , G , F ,  Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling )
            initf = norm( W.*(X-G*F) , 'fro' )^2;
            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                Result=WX+ (I-W).*(G*F);
                for j=1:Iter_max_M_step
                    %M Step
                    if(Scaling)
                        Result=[Result;15*ones(1,n2)]; G=[G;15*ones(1,reducedDim)];
                        [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'SCALLING',false,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                        Result=Result(1:end-1,:);
                        G=G(1:end-1,:);
                    else
                        %                         [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                        I_HS=reshape(Result',[sqrt(n2),sqrt(n2),n1]);
                        abundanceMap = estimateAbundanceLS(I_HS,G,'Method','ncls');
                        F=reshape(abundanceMap,[sqrt(n2)*sqrt(n2),reducedDim]);
                        F=F';
                    end
                end
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            f = norm( W.*(Result-G*F) , 'fro' )^2;


        end
        %%
        function [Result , G, F , t , initf , f] = EM_L1R_WNE_NMF( W ,X , G , F , Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling ,beta )
            initf = norm( W.*(X-G*F) , 'fro' )^2;
            r=size(G,2);
            [m,n]=size(W);
            Ginit = round(255.*rand(m,r))+0.00001;
            Finit = rand(r,n)+0.00001;
            initfTemp = norm( W.*(X-Ginit*Finit) , 'fro' )^2;

            if(initf>initfTemp)
                G=Ginit;
                F=Finit;
            end

            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);
            if(Scaling)
                %                F=ScaleColumns(F);
                G=ScaleRows(G);
            end
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                Result=WX+ (I-W).*(G*F);
                %               Result=(I-W).*X+ (I).*(G*F);

                for j=1:Iter_max_M_step
                    %M Step
                    [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'TYPE','L1R','BETA',beta,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);

                end
                if(Scaling)
                    G=ScaleRows(G);
                    %                    F=ScaleColumns(F);
                end
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );
        end

        %%
        function [Result , G, F , t , initf , f] = EM_L2R_WNE_NMF( W ,X , G , F  ,Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling,beta )
            initf = norm( W.*(X-G*F) , 'fro' )^2;
            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);
            %             F=ScaleColumns(F);
            if(Scaling)
                G=ScaleRows(G);
            end;
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                Result=WX+ (I-W).*(G*F);
                for j=1:Iter_max_M_step
                    %M Step
                    [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'TYPE','L2R','BETA',beta,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);

                end
                %                 F=ScaleColumns(F);
                if(Scaling)
                    G=ScaleRows(G);
                end
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f )
        end

        %%
        function [Result , G, F , t , initf , f] = EM_GWNE_NMF( W ,X , G , F , Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations ,S,beta ,Scaling)
            %S is the Adjacency Matrix of the graph, its dimensions must be
            %Total_Number_Of_Pixels x Total_Number_Of_Pixels
            %             Temp=G;
            %             G=F';
            %             F=Temp';
            %             W=W';
            %             X=X';
            initf = norm( W.*(X-G*F) , 'fro' )^2;
            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);
            %             F=ScaleColumns(F);
            if(Scaling)
                G=ScaleRows(G);
                %                 F=ScaleColumns(F);
            end
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                Result=WX+ (I-W).*(G*F);
                for j=1:Iter_max_M_step
                    %M Step
                    [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'TYPE','MR','S_MTX',S,'BETA',beta,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                end
                %                F=ScaleColumns(F);
                if(Scaling)
                    G=ScaleRows(G);
                    %                 F=ScaleColumns(F);
                end;
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );
            %             Temp=G;
            %             G=F';
            %             F=Temp';
            %             Result=Result';
        end

        %%
        function [Result , G, F , t , initf , f] = EM_SPA_WNMF( W ,X , G , F ,  Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling )
            initf = norm( W.*(X-G*F) , 'fro' )^2;


            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);

            if(Scaling)
                G=ScaleRows(G);
            end
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                fprintf('\n ### Inside E step , iteration number = %d ...',i);
                Result=WX+ (I-W).*(G*F);
                for j=1:Iter_max_M_step
                    %M Step
                    %                   [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                    [F , G ,it,ela,HIS]=SPA(Result,reducedDim,G,F);
                    F=F';
                    G=G';
                end
                if(Scaling)
                    G=ScaleRows(G);
                end
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );

        end
        %%
        function [Result , Ginit, F , t , initf , f] = EM_WNE_NMF_Fixed_F( W ,X , Ginit , F ,  Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,alpha,gamma,beta )
            initf = norm( W.*(X-Ginit*F) , 'fro' )^2;
            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(Ginit,2);


            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                fprintf('\n ### Inside E step , iteration number = %d ...',i);
                Result=WX+ (I-W).*(Ginit*F);
                for j=1:Iter_max_M_step
                    %M Step
                    %                   [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                    Result=[Result;alpha*ones(1,n2)]; Ginit=[Ginit;gamma*ones(1,reducedDim)];
                    %                     [ G , F , iter,elapse,HIS]=NeNMF_Fixed_W(Result,reducedDim,'MAX_ITER',1000,'MIN_ITER',10,'W_FFIXID',Ginit,'H_INIT',F);
                    [ G , F , iter,elapse,HIS]=NeNMF_Fixed_W(Result,reducedDim,'TYPE','L1R','BETA',beta,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations,'W_FFIXID',Ginit,'H_INIT',F);
                    %
%                     [F,G] = NMF_entropy_adaptive(Result, reducedDim, 1000, 0.02,Ginit,F);

%                     [ G , F ]=MU_NMF_With_Entropy( Result , Ginit , F ,1000, 0.00000000000000001 );
%                      [~,F]=simplex_project(Result,Ginit);
%                      F=reshape(abundanceMap,[sqrt(n2)*sqrt(n2),reducedDim]);

                    Result=Result(1:end-1,:);
                    Ginit=Ginit(1:end-1,:);
                end
            end
            t = 10;
            f = norm( W.*(Result-Ginit*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );

        end
        %%
        function [Result , G, F , t , initf , f] = EM_PPI( W ,X , G , F ,  Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling )
            initf = norm( W.*(X-G*F) , 'fro' )^2;


            WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);

            if(Scaling)
                %                G=ScaleRows(G);
                G=Scale(G',6)';
                %                F=ScaleRows(F);
                %                 F=ScaleColumns(F);
            end
            %E Step
            Result=[];
            %             Starting_Matrix=repmat(G,1,size(X,2));
            for i=1:Iter_max_E_step
                fprintf('\n ### Inside E step , iteration number = %d ...',i);
                Result=WX+ (I-W).*(G*F);
                I_HS=reshape(Result',[sqrt(n2),sqrt(n2),n1]);
                for j=1:Iter_max_M_step
                    G=ppi(I_HS,reducedDim, 'ReductionMethod','PCA');
                    %                     block_iteration=1;
                    %                     a=1;
                    %                     b=100;
                    %                     max_iter=size(Result,2)/100;
                    %                     for rn=1:max_iter
                    %                         Sun_D{block_iteration}=Result(:,a:b);
                    %                         a=a+100;
                    %                         b=b+100;
                    %                         block_iteration=block_iteration+1;
                    %                     end
                    %
                    %                     Sun_R=cell(numel(Sun_D),1);
                    %                     lambda = 2e-3;
                    %                     parfor tt=1:numel(Sun_D)
                    %                         M=Sun_D{tt};
                    %                         fprintf('Running Sunsal patch %d',tt);
                    %                         Sun_R{tt} = sunsal(M,G,'lambda',lambda,'ADDONE','yes','POSITIVITY','yes', ...
                    %                                 'AL_iters',200,'TOL', 1e-4, 'verbose','yes');
                    %                     end
                    %
                    %                     G_Final_Sunsal=[];
                    %                     for tt=1:numel(Sun_R)
                    %                         M=Sun_R{tt};
                    %                         G_Final_Sunsal=[G_Final_Sunsal;M];
                    %                     end
                    %                     F=G_Final_Sunsal';
                    F=estimateAbundanceLS(I_HS,G,'Method','ncls');
                    F=reshape(F,[reducedDim,n2]);
                    %                     [ Gdd , F , iter,elapse,HIS]=NeNMF_Fixed_W(Result,reducedDim,'MAX_ITER',1000,'MIN_ITER',10,'W_FFIXID',G,'H_INIT',F);
                    %                   %M Step
                    %                   [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                end
                %               F=ScaleColumns(F);
                %             G=ScaleColumns(G);
                %             F=ScaleRows(F);
                if(Scaling)
                    %                G=ScaleRows(G);
                    G=Scale(G',6)';
                    %                 F=ScaleColumns(F);
                end
                if(Iter_max_M_step>3)
                    Iter_max_M_step=Iter_max_M_step-1;
                end
            end
            t = 10;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );

        end

        %%
        % Weighted NMF with updated weights
        function [Result , G, F , t , initf , f] = EM_WEucNMF( W ,X , G , F ,  Iter_max_M_step , Iter_max_E_step , Nesterov_Max_Iterations , Nesterov_Min_Iterations,Scaling ,beta)
            initf = norm( W.*(X-G*F) , 'fro' )^2;

            %             WX = W.*X;
            [n1,n2]=size(W);
            I=ones(n1,n2);
            reducedDim=size(G,2);
            if(Scaling)

            end
            %E Step
            Result=[];
            for i=1:Iter_max_E_step
                %                 fprintf('\n ### Inside E step , iteration number = %d ...',i);
                WX = W.*X;
                Result=WX+ (I-W).*(G*F);
                for j=1:Iter_max_M_step
                    %M Step
                    if(Scaling)
                        Result=[Result;1*ones(1,n2)]; G=[G;1*ones(1,reducedDim)];
%                         [G , F ,it,ela,HIS]=NeNMF(Result,reducedDim,'SCALLING',false,'W_INIT',G,'H_INIT',F,'MAX_ITER',Nesterov_Max_Iterations,'MIN_ITER',Nesterov_Min_Iterations);
                        [T, ~ , F,it]=WEucNMF(Result,G,F,Nesterov_Max_Iterations,beta);
                        Result=Result(1:end-1,:);
                        G=G(1:end-1,:);
                    else

                        [T, ~ , F ,it]=WEucNMF(Result,G,F,Nesterov_Max_Iterations,beta);

                    end
                end
                %                 W=T;

            end
            t = 10;
            %             f = norm( W.*(Result-G*F) , 'fro' )^2;
            G_temp=ones(n1,reducedDim);
            %             f = norm( W.*(Result-G_temp*F) , 'fro' )^2;
            f = norm( W.*(Result-G*F) , 'fro' )^2;
            %             fprintf('\n### Elapse time: %d sec.\n###   Initial objective value: %d\n###   Objective value: %d \n' , t , initf , f );

        end
    end
end
