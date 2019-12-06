function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%vectorized implementation of finding cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


h=sigmoid(X*theta); % m x (n+1) and (n+1)x1... additional "1" to account for intercept

%finding the term which accounts for regularization
reg_term = (lambda / (2*m) )* ( (theta'*theta) - theta(1,1)^2 ) ;

J= (1/m)* ( -y'* log(h) - (1-y)'*log(1-h) ) + reg_term ;

%%%%%%%%%%% End of above implementation %%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%vectorized implementation of gradient of cost function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%finding the term which accounts for regularization
reg_term_grad = (lambda/m)*theta ; 

%finding gradient for all terms even the intercept term (for which we will  
grad = (1/m) * ( X'*( sigmoid(X*theta)- y ) ) + reg_term_grad ;

%for constant intercept term we are not including the regularized term
grad_inter= (1/m)*  ( X'*( sigmoid(X*theta)-y) ) ;

grad(1,1)=grad_inter(1,1);

%%%%%%%%%%% End of above implementation %%%%%%%%%%%



% =============================================================

grad = grad(:);

end
