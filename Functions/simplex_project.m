function [y, a] = simplex_project(x, E, D)
% SIMPLEX_PROJECT Project point onto simplex
%
% Syntax:
% [y, a] = simplex_project(x, E, D)
%
% This function projects the N-dimensional points in x onto the
% simplex spanned by the p points in E, and returns the projected points 
% and their barycentric coordinates with respect to the points in E. The
% projection is defined as the point within the simplex with minimal
% L2-distance.
% This is also the solution to the fully constrained least-squares problem
% x=E*a, with positivity and sum-to-one constraints imposed on a.
% 
% The algorithm is NOT guaranteed to find the optimal/true solution in all
% cases.
% 
% The algorithm used is described in 
%    Heylen R., Burazerovic D., Scheunders P., "Fully Constrained Least 
%    Squares Spectral Unmixing by Simplex Projection," Geoscience and
%    Remote Sensing, IEEE Transactions on, 49(11), 4112-4122, 2011.
%    doi: 10.1109/TGRS.2011.2155070
%    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5887410&isnumber=4358825
%
% Input:
% x : NxM vector, the M points to project
% E : Nxp matrix, the p points spanning the simplex
% D : (optional) pxp squared distance matrix of the endmembers
%
% Output:
% y : NxM vector, the projected points
% a : pxM vector, barycentric coordinates of y, so that y = E*a
%
%
% IMPORTANT NOTE: 
% For numerical stability, the input variables should not be too large. All
% inputs can for instance be rescaled to the interval [0,1] before use.
%
% Author: Rob Heylen
% Visionlab, University of Antwerp, January 2011


% Obtain matrix sizes
[N,M]=size(x);
p=size(E,2);

% The trivial zero-dimensional problem: projection onto a single point.
if p==1
    a=ones(1,M);
    y=E*a;
    return;
end

% Project points onto simplex plane, obtain barycentric coordinates.
x_orig=x;
[x,a]=plane_project(x,E);

% Test positivity and sum-to-one of barycentric coordinates, AND both
% tests.
d1=(sum(a(2:p,:)>=0,1))==(p-1); 
d2=sum(a(2:p,:),1)<=1;
d=d1.*d2; 

% Turn d into a list of all points not inside the simplex.
d=find(~d);
Md=length(d);

% If all points are inside the simplex, finish here
if Md==0
    y=E*a;
    return;
end

% Put the barycentric coordinates of points not in the simplex to zero.
a(:,d)=zeros(p,Md);

% We will need the squared distance matrix. If not yet given, calculate it.
if nargin==2
    D=squareform(pdist(E')).^2;
end

% Calculate the incenter via its barycentric coordinates.
ac=zeros(p,1);
for i=1:p
    range=[1:i-1 i+1:p];
    C_face=[D(range,range) ones(p-1,1); ones(1,p-1) 0];
    ac(i)=sqrt(det(C_face)*(-1)^(p-1));
end
c=E*(ac/sum(ac));

% Translation to put the incenter at the origin.
xc=x(:,d)-c*ones(1,Md);
Ec=E-c*ones(1,p);

% Loop over all vertices
for vertex=1:p
    range=[1:vertex-1 vertex+1:p];
    
    % Determine the points in the bissective cone of this vertex
    b=linsolve(Ec(:,range),xc);
    bi=prod(real(b>0),1);
    zi=find(bi); 

    % If there are points in this cone, recursive call
    if ~isempty(zi)
        xr=x_orig(:,d(zi));
        Er=E(:,range);
        [y,ar]=simplex_project(xr,Er,D(range,range));
        a(range,d(zi))=ar;
        
        % Remove treated points from list d
        zd=setdiff(1:Md,zi);
        d=d(zd);
        xc=xc(:,zd);
        Md=length(d);
        
    end
    
    if Md==0
        break; 
    end
    
end

% It is possible that due to numerical issues, the list d is not empty
% (e.g. a point that lies exactly on the boundary between two cones). These
% points could be treated here with an extra routine (still to do...).

% Optionally we could include a check here to see whether all the
% projections are correct. The points that were not correctly projected
% could then be treated via an alternative algorithm.

y=E*a;

end



function [y,a]=plane_project(x,E)
    [N,M]=size(x);
    p=size(E,2);
    a=zeros(p,M);
    ct=E(:,1);
    Ep=E(:,2:p)-ct*ones(1,p-1);
    a(2:p,:)=Ep\(x-ct*ones(1,M));
    a(1,:)=ones(1,M)-sum(a(2:p,:),1);
    y=E*a;
end

