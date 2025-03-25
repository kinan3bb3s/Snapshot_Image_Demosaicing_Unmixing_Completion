function [G] = AbundanceSuperPixelIdeal(I_MOS_seq,num_band,F_WNMF,alg,rank)
%ABUNDANCESUPERPIXEL Summary of this function goes here
%   Detailed explanation goes here
Cube=X2Cube(I_MOS_seq,5,25);
[n1,n2]=size(I_MOS_seq);
% Cube=applyCorrectionMatrix(Cube,num_band+1);


F=F_WNMF;
layer=20*ones(size(Cube,1),size(Cube,2));
I_WNMF_rec2=Cube;
I_WNMF_rec2(:,:,num_band+1)=layer;
Final_F2=F;
Final_F2(:,end+1)=30*ones(rank,1);
abundanceMap3 = estimateAbundanceLS(I_WNMF_rec2,Final_F2','Method','ncls');
G_New4=abundanceMap3;
G_New4(G_New4<0)=0;

% temp=reshape(G_New4,[size(Cube,1)*size(Cube,2),rank]);
% A=reshape(Cube,[size(Cube,1)*size(Cube,2),num_band]);
% A1=[A,20*ones(size(A,1),1)]; Final_F1=[F_WNMF,20*ones(rank,1)];
% [ ~ , G_New4 , ~,~,~]=NeNMF_Fixed_W(A1',rank,'MAX_ITER',1000,'MIN_ITER',100,'W_FFIXID',Final_F1','H_INIT',temp','SCALLING',false);
% G_New4=reshape(G_New4,[size(Cube,1),size(Cube,2),rank]);

% alg='lanczos3';%bicubic,lanczos3,nearest,bilinear
for i=1:rank
    G_New5(:,:,i)=imresize(G_New4(:,:,i),[n1,n2],alg);
end
G=reshape(G_New5,[n1*n2,rank]);
G(G<0)=0;
for i=1:size(G,1)
    temp=G(i,:);
    for j=1:size(temp,2)
        if(temp(j)<0.1)
            temp(j)=0;
        end
    end

    G(i,:)=temp;
end
G=ScaleRows(G);
% Image=G*F;
% A1=[Image,1*ones(size(Image,1),1)]; Final_F1=[F_WNMF,1*ones(rank,1)];
% [ ~ , G , ~,~,~]=NeNMF_Fixed_W(A1',rank,'MAX_ITER',1000,'MIN_ITER',10,'W_FFIXID',Final_F1','H_INIT',G','SCALLING',false);
% G=G';
end

