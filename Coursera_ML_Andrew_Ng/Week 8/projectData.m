function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% extracting the top K eigen vectors from matrix U
U_reduce = U( :, 1:K ) ; ... since U is nxn, so U_reduce has dimension nxK

% finding reduced representation of m examples from R^n to R^K sub-space
Z = X * U_reduce ;  ... X is mxn and U_reduce is nxK.. so Z is mxK.. i.e.
                    ... we have m observations which have been reduced to K-dimesnions




% =============================================================

end
