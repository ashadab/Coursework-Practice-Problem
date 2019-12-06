function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


%finding unregularized cost function
prediction = X*Theta'; 

deviation = prediction - Y ; 

J_inter = ( deviation .* deviation) .* R ;

%creating variables for finding regularized cost function
theta_inter = Theta.^2 ; 
X_inter = X.^2 ; 

J = (1/2) * sum ( J_inter(:) ) + 0.5*lambda * sum( theta_inter(:) ) + 0.5*lambda * sum( X_inter(:) ) ;

%% finding unregularized gradient for X and Theta %%%%

%finding index of the elements for which a movie is not rated by a particular
... user and hence value is equal to zero in the R matrix
id = find(R==0) ; 

%creating a variable for which deviation values should be equal to zero
... if there movie is not rated by a particular user
deviation_inter  = deviation ; 

deviation_inter(id) = 0 ; 

% deviation_inter is n_m x n_u and Theta is n_u x n
X_grad =  ( deviation_inter * Theta )  + (lambda)* X ; 

% deviation_inter' is n_u x n_m and X is n_m x n
Theta_grad = ( deviation_inter' * X )  + (lambda)* Theta ; 







% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
