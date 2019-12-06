function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%% computing cost
... note that in ex5.m when linearRegCostFunction is called, the code introduces a column of 
... ones in the matrix X so its dimesions changes from mxk to mx(k+1)

% we will run below commented out code if the vector of ones 
... is NOT introduced in the code. 
... X_bar = [ ones(m,1) X] ; 
... h_x = X_bar * theta ; 


%finding prediction terms
h_x = X * theta ;  % X is mx(k+1)... theta is (k+1)x1.. hence h_x is mx1

%computing the term used in the regularization part
sum_reg_term = ( theta(2:end,1) )' * ( theta(2:end,1) ) ; ... theta(2:end,1) is kx1

%computing the term used in the error deviation part
deviation_term = (h_x - y)' * (h_x - y) ;

% computing the cost
J = ( 1/(2*m) ) * deviation_term + ( lambda/(2*m) ) * sum_reg_term ; 


%% computing gradient

% No need of defining grad again since it is already defined at the start
... of this function
... grad = zeros( size(theta,1),1 ) ;  


% implementing a vectorized version of gradient
... note that in ex5.m when linearRegCostFunction is called, the code introduces a column of 
... ones in the matrix X so its dimesions changes from mxk to mx(k+1)
... X' is (k+1)x m where k is #features; (h_x-y) is mx1 and theta is (k+1)x1
grad = (1/m)* X'*(h_x - y) + (lambda/m) * theta ; 

% updating value of gradient vector ... X(:,1) is mx1 and (h_x-y) is also mx1
grad(1) = (1/m)* ( X(:,1) )'* (h_x - y) ; 



%% =========================================================================

grad = grad(:);

end
