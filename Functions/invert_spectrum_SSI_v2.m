function [X] = invert_spectrum_SSI_v2(Y,alpha,number_of_selected_wavelength,A,regluarization,Entropy_A,Window_size)
% Y is the simulated spectrum
% alpha is the regularization parameter
% number_of_selected_wavelength must equal the number of the virtual
% wavelengths
% A is the sampled response matrix
% regluarization is the type of the regluarization


if(regluarization=="Tek")
    %Creat the D matrix for tikhnov regularization
    D=eye(number_of_selected_wavelength,number_of_selected_wavelength);
    for i=1:(number_of_selected_wavelength-1)
        for j=1:number_of_selected_wavelength
            if(i==j)
                D(i,j+1)=-1;
            end
        end
    end

    %      D=2*eye(number_of_selected_wavelength,number_of_selected_wavelength);
    %     for i=1:(number_of_selected_wavelength-1)
    %         for j=1:number_of_selected_wavelength
    %             if(i==j)
    %                 D(i,j+1)=-1;
    %                 if(j>1)
    %                     D(i,j-1)=-1;
    %                 end
    %             end
    %         end
    %     end
    %     D(number_of_selected_wavelength,1)=-1;
    D(number_of_selected_wavelength,number_of_selected_wavelength)=0;
    %     D(1,1)=1;
    %     D(1,2)=0;
    e=zeros(number_of_selected_wavelength,1);
    Y=[Y;e];

    % Modifiy the A matrix for tikhnoe regularization
    A=[A;alpha*D];

    % Solve the optimization problem
    X=lsqnonneg(A,Y);
elseif('cvx')
    D=eye(number_of_selected_wavelength,number_of_selected_wavelength);
    for i=1:(number_of_selected_wavelength-1)
        for j=1:number_of_selected_wavelength
            if(i==j)
                D(i,j+1)=-1;
            end
        end
    end
%     D(number_of_selected_wavelength,1)=-1;
D(number_of_selected_wavelength,number_of_selected_wavelength)=0;
% %     D(number_of_selected_wavelength,number_of_selected_wavelength)=0;
%     D(1,1)=1;
%     uu=0.4;
%         
%         D(46,46)=uu;
%         D(47,47)=uu;
%         D(48,48)=uu;
%         D(49,49)=uu;
%         D(50,50)=uu;
%         D(51,51)=uu;
%         D(52,52)=uu;
%         D(53,53)=uu;
%         D(54,54)=uu;
    %      D(number_of_selected_wavelength,1)=-1;
    %     D(1,:)=1/number_of_selected_wavelength;
    
    cvx_begin quiet
        variable X(number_of_selected_wavelength)
      
        minimize( norm( A*X-Y ) + alpha*norm(Entropy_A*D*X,2))
%         minimize( norm( A*X-Y ) + alpha*norm(D*X,2))
        subject to
        0<=X
    cvx_end

elseif (regluarization=="L2")
    % L2 regularization goes here
else
    X=lsqnonneg(A,Y);
end
% window_size=30;
if Window_size>0
    X = movmean(X,Window_size);
end
% X=smoothdata(X);
X=X';
end

