function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    
    % declaring some initial variables
    
    sum_theta_1=0;
    sum_theta_2=0;
    
%     for k= 1:length(theta)
%         sum_theta=0 ;
%     end
    
    
    % finding sum of deviations across all observations

    for i= 1:m
        
        %creating sum for intercept term 
        sum_theta_1 = sum_theta_1 + ( (theta.' * X(i,:).') - y(i,1) )    ;
         
        
        %creating sum for slope term 
        sum_theta_2 = sum_theta_2 + ( (theta.' * X(i,:).') - y(i,1) ) * X(i,2) ;   
        
              
    end

% doing simultaneuous updates to theta   
theta_1 = theta(1,1) - (alpha/m)*sum_theta_1;
theta_2 = theta(2,1) - (alpha/m)*sum_theta_2;

% finding final values of theta so that it could be used in next iteration
theta=[theta_1; theta_2];



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end %#iternation loop end

end %function's end
