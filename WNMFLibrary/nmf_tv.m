function [W, H] = nmf_tv(V, k, lambda_tv, max_iter, tol, W_init, H_init)
% NMF with TV regularization using multiplicative updates
% Inputs:
% V: input data matrix of size (m x n)
% k: number of endmembers
% lambda_tv: regularization parameter for TV penalty
% max_iter: maximum number of iterations
% tol: tolerance for convergence
% W_init: initial endmembers matrix of size (m x k)
% H_init: initial abundance matrix of size (k x n)
% Outputs:
% W: estimated endmembers matrix of size (m x k)
% H: estimated abundance matrix of size (k x n)

[m, n] = size(V);
epsilon = 1e-10;  % small value added to denominator to avoid division by zero

% Initialize matrices W and H
W = W_init;
H = H_init;

% Define finite difference matrix
D = spdiags([-ones(n,1), ones(n,1)], [0,1], n-1, n);
D(:,end) = 0;

% Define auxiliary variables P and G
P = D * H;
G = sqrt(P.^2 + epsilon);

% Iterative update of W and H
for i = 1:max_iter
    % Update H
    numerator = W' * V + lambda_tv * D' * (P./(G));
    denominator = W' * W * H + lambda_tv * D' * ones(size(G));
    H = H .* (numerator ./ denominator);
    
    % Update W
    numerator = V * H' + lambda_tv * D * (P./(G));
    denominator = W * (H * H') + lambda_tv * ones(m, k);
    W = W .* (numerator ./ denominator);
    
    % Compute residual error and check for convergence
    err = norm(V - W * H, 'fro') / norm(V, 'fro');
    if err < tol
        break
    end
    
    % Update auxiliary variables P and G
    P = D * H;
    G = sqrt(P.^2 + epsilon);
end
end
