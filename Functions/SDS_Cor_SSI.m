function I_HS=SDS_Cor_SSI(I,SpectralProfiles_Full,wavelengths,alpha)

A=SpectralProfiles_Full(:,wavelengths-400);
A=ScaleRows(A);
Entopy_A=calcEntropyDiag(A');

[n1,n2,n3]=size(I);
I_HS=zeros(n1,n2,n3-1);
window_size=0;
number_of_virtual_wavelength=size(wavelengths,2);
for i=1:n1
    parfor j=1:n2
        tmp=I(i,j,:);
        tmp=tmp(:);
        fprintf('Spectral correction for pixel (%d,%d) \n',i,j);
        corrected=invert_spectrum_SSI_v2(tmp,alpha,number_of_virtual_wavelength,A,'cvx',Entopy_A,window_size);

        I_HS(i,j,:)=corrected;
    end
end

end