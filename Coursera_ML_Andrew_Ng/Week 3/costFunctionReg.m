function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%vectorized implementation of finding cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


h=sigmoid(X*theta); % m x (n+1) and (n+1)x1

%finding the term which accounts for regularization
reg_term = (lambda / (2*m) )* ( (theta'*theta) - theta(1,1)^2 ) ;

J= (1/m)* ( -y'* log(h) - (1-y)'*log(1-h) ) + reg_term ;

%%%%  semi-vectorized version %%%%

% sum=0;
% 
% for i= 2:size(theta,1)
%     
%     sum = sum + ( theta(i,1)^2 ) ;
%     
% end
% 
% J= (1/m)* ( -y'* log(h) - (1-y)'*log(1-h) ) + ( 1/ (2*m) )* lambda * sum ;


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


%%%%  semi-vectorized version %%%%

%grad = X'*( sigmoid(X*theta)-y);
%grad = (1/m)*grad ; 

% for i=1:m
%     
%     grad(1,1)= (1/m)* ( X'*( sigmoid(X*theta)-y ) ) ;  % 
%     
%    
% end




% =============================================================

end
