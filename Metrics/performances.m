% Computes the performance of endmember extraction w.r.t 
% to true spectral signatures H
% 
% Also return sorted and scaled V

function [Vout, perf] = performances(V, H); 

r = size(H,1); 

% Compute the MRSA
for i = 1 : r
    for j = 1 : r
        h = H(i,:) - mean(H(i,:)); 
        v = V(j,:) - mean(V(j,:)); 
        Dist(i,j) = acos( h*v'/norm(v)/norm(h)  ); 
    end
end
Dist = Dist/pi; 
[assignment,cost] = munkres(Dist); 
for i = 1 : r 
    Vout(i,:) = V(assignment(i),:); 
    perf(i) = 100*Dist(i,assignment(i)); 
end