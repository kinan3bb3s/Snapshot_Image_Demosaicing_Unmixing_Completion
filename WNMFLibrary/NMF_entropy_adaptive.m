function [F,G] = NMF_entropy_adaptive(X, rank, max_iter, eta,G,F)
%     % Initialize the factor matrices G and F
%     G = rand(size(X,1), rank);
%     F = rand(rank, size(X,2));

    % Define the entropy function
    entropy = @(F) -sum(F.*log(F+1e-9),1);

    % Define the adaptive alpha function
    alpha = @(entropy, eta) max(0.1, eta*(1-entropy/log(size(F,1))));

    % Define the stopping criteria
    tolerance = 1e-6;
    
    % Start the iterative updates
    for i = 1:max_iter
        % Update F using the multiplicative update rule with adaptive alpha
        F_prev = F;
        F = F .* (G'*X) ./ (G'*G*F + alpha(entropy(F), eta));

        % Update G using the multiplicative update rule 
%         G = G .* (X*F') ./ (G*F*F');

        % Check for convergence
        if norm(F - F_prev, 'fro')/norm(F_prev, 'fro') < tolerance
            break;
        end
    end
end