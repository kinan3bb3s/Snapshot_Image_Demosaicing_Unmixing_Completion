function F_HS=SDS_Cor_SSI_Endmemebers(F,SpectralProfiles_Full,wavelengths,alpha)

A=SpectralProfiles_Full(:,wavelengths-400);
A=ScaleRows(A);
Entopy_A=calcEntropyDiag(A');

[n1,n2]=size(F);
window_size=0;
F_HS=zeros(n1,n2-1);
number_of_virtual_wavelength=size(wavelengths,2);
for i=1:n1
        tmp=F(i,:);
        tmp=tmp(:);
        fprintf('Spectral correction for pixel (%d) \n',i);
        corrected=invert_spectrum_SSI_v2(tmp,alpha,number_of_virtual_wavelength,A,'cvx',Entopy_A,window_size);

        F_HS(i,:)=corrected;
    end
end

