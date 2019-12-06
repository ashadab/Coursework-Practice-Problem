function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %number of features/predictors in the dataset including the constant 1 columns
    nvar = size(X,2); 
    
    % creating a row vector which will have sum across all the variables
    sum_theta= zeros(1, nvar);
    
    for i= 1:m
        
    %finding sum of deviations for intercept term
    sum_theta(1)= sum_theta(1) + ( (theta.' * X(i,:).') - y(i,1) )    ;
    
    end
    
    
         
    %finding sum of deviations for slope terms
    %nvar = size(X,2); %number of features in the dataset
    %nvar1 = nvar+1;
    
    for j= 2: nvar 
        
        for i= 1:m
            
            %creating sum for slope term 
            sum_theta(j) = sum_theta(j) + ( (theta.' * X(i,:).') - y(i,1) ) * X(i,j) ;   
        
        end
     
    end
    
    
        % doing simultaneuous updates to theta   
        theta_inter=zeros(nvar,1);
        
        for k=1:nvar
            
            theta_inter(k)= theta(k,1)- (alpha/m)*sum_theta(k);
        
        end
        

% finding final values of theta so that it could be used in next iteration
theta = theta_inter.' ; %taking transpose of theta_inter (which is a row vector) since theta is a column vector


    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
