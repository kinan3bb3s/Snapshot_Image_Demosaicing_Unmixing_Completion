function [ G , F ] = MU_NMF_With_Entropy( X , G_Fixed , F , Iter_max, alpha )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                       %
%   NMF using Lee & Seung Multiplicative update rules   %
%                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                       %
%  solve:    argmin_G,F || X - G.F ||^2_Fro             %
%                                                       %
%            s.t. G , F >= 0                            %
%                                                       %
%  input: X = m*n matrix of data                        %
%         Ginit = m*r matrix (initialization for G)     %
%         Finit = r*n matrix (initialization for F)     %
%                                                       %
%  output: estimated G and F                            %
%                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   author : Clément Dorffer                            %
%   date : 15/05/2016                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initf = norm( X-G*F , 'fro' )^2;
% tic
for i = 1 : Iter_max    
    % updating G
%     G = Updt_G( X , G , F );  
%     entropy = -sum(X.*log(X+1e-9),1);
%     alpha = @(entropy) max(0.1, 1 - entropy);

    G=G_Fixed;
    % updating F
    F = Updt_F( X , G , F, alpha );  
%     F = F.*((G'*X + alpha(entropy).*ones(1,size(X,2)))./(G'*G*F - alpha(entropy).*log(F)));
end
% t = toc;
% f = norm( X-G*F , 'fro' )^2;
% fprintf('\n###   Elapse time: %d sec.\n###   Initial objective value: %d.\n###   Objective value: %d \n' , t , initf , f );
end



function [ F ] = Updt_F( X , G , F , alpha)
%%%%%%%%%%%%%%%%%%%%%
% update rule for F %
%%%%%%%%%%%%%%%%%%%%%%
F = F.*((G'*X + 1*alpha)./secu_plus(G'*G*F - alpha.*log(F),eps));
% F = F.*((G'*X + alpha(entropy)*ones(1,size(X,2)))./(G'*G*F - alpha(entropy)*log(F)));
end



function [ G ] = Updt_G( X , G , F )
%%%%%%%%%%%%%%%%%%%%%
% update rule for G %
%%%%%%%%%%%%%%%%%%%%%
G = G.*((X*F')./secu_plus(G*(F*F'),eps));
end



function [toto] = secu_plus(tutu,s)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Matthieu Puigt
% Date: 20/04/2015
% Goal: Security in the NMF procedure which project the negative data to
% epsilon (small user-defined threshold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(nargin<2)
    s=1.e-8;
end
toto=max(tutu,s);
tutu(isnan(tutu)) = 0;
end
