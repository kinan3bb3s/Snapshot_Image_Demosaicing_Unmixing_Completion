function [y] = applyCorrectionMatrix(I,num_band)
%APPLYCORRECTIONMATRIX Summary of this function goes here
%   Author Kinan ABBAS
%   Creation Date: 26 OCT 2022


if(num_band==25)
    load('Data\correction_matrix.mat')
else
    load('Data\Xemia_16_corrected.mat')
end
 for i=1:size(I)
    for j=1:size(I,2)
        tmp=I(i,j,:);
        tmp=tmp(:);
        corrected=correction_matrix*tmp;
        y(i,j,:)=corrected;
    end
end
end

