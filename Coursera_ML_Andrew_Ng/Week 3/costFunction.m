function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%



% %finding cost function
% 
% sum_terms=0;
% 
% for i=1:m
%     
%     % theta'*X can be written as ( 1X(n+1) * (n+1)X1 ) 
%     z= X(i,:)* theta ;
%     
%     %defining hypothesis function 
%     hypo_function = sigmoid(z) ; 
% 
%     sum_terms= sum_terms + ( -y(i,1)*log(hypo_function) ) - ( (1-y(i,1))*log(1-hypo_function) ) ;
%     
% end
% 
% %final step in computing cost function
% J= (1/m)* sum_terms;


%vectorized implementation of finding cost

h=0;
h=sigmoid(X*theta);
J= (1/m)* ( -y'* log(h) - (1-y)'*log(1-h) ) ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% computing gradient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%finding component wise gradient for each term

% for j=1: size(theta,1)
% 
%         for i=1:m
%             
%             % theta'*X can be written as ( 1X(n+1) * (n+1)X1 ) 
%             z= X(i,:)* theta ;
%             
%             %defining hypothesis function 
%             hypo_function = sigmoid(z) ; 
%             
%             
%             %writing vectorized version for gradient formulae 
%             grad(j,1) = grad(j,1) +  (hypo_function-y(i,1)) * X(i,j);
%             
%             
%             
%         end
%         
%         %final step in computing gradient
%         grad(j,1) = (1/m)* grad(j,1) ; 
% 
% end


%vectorized implementation of gradient

grad = X'*( sigmoid(X*theta)-y);
grad = (1/m)*grad ; 

% =============================================================

end
