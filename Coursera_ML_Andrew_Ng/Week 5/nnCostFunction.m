function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Part 1 %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%

%finding number of training examples
m = size(X,1);

%finding number of input features
n = size(X,2);

J_inter = 0; 

size (X(m,:)',1 );

%finding number of unique labels in output y 
label_size = size(unique(y),1) ;


% assigning value equal to 1 to the component of y_vector
    ... whose value matches with the outcome in y variable 
y_vector =zeros(label_size,m);

for j= 1:m

    y_vector ( y(j) , j   ) = 1 ;
    
end


% we find cost for each training example and then sum it across all the
... training examples

for inp_dat= 1:m
    
    % layer 1 (input layer) computation for a single training example
    a_1 = [ 1 ; X(inp_dat,:)'] ; % we add 1 for the bias unit
    size(a_1,1);
    
    % layer 2 (hidden layer) computation for a single training example
    z_2 = Theta1 * a_1 ; 
    a_2 = [1; sigmoid(z_2)] ; 
    
    % layer 3 (output layer) computation for a single training example
    z_3= Theta2 * a_2 ; 
    a_3 = sigmoid(z_3) ; 
    h_x = a_3 ; ... its dimensions are kx1
        
    % initializing a vector which will capture cost across all the
    ... K-classes 
    p=zeros(label_size,1) ; 
    
    y_train_ex = y_vector(:,inp_dat) ; ... its dimensions are kx1
       
    % finding sum across all the class labels
    for k= 1:label_size
        
        p(k,1) = - y_train_ex(k,1) * log( h_x(k,1) ) - ( 1 - y_train_ex(k,1) ) * log( 1- h_x(k,1) ) ; 
        
    end
    
    % finding intermediate value of cost
    J_inter = J_inter + sum(p);

end


% removing th 1st column which corresponds to the bias term since 
... we don't want to regularize the bias weight /parameter 
Theta1_new = Theta1( : , 2:size(Theta1,2)) ;
Theta2_new = Theta2( : , 2:size(Theta2,2)) ;

% finding square of each element of the matrix 
Theta1_reg = Theta1_new.^2 ;
Theta2_reg = Theta2_new.^2 ;

% finding sum of ALL elements of the matrix so that it could we used in
... regularization
sum_Theta1_reg = sum(Theta1_reg,"all") ; 
sum_Theta2_reg = sum(Theta2_reg,"all") ; 

%computing final value of the regularized cost.. when lamda=0 we get the
... normal cost function WITHOUT regularization
    
J = (1/m) * J_inter +  ( (lambda)/(2*m) ) * ( sum_Theta1_reg + sum_Theta2_reg);




%% ======= Part 2 : Implementing Backpropogation algorithm =======

... we find gradient for each training example and then accumalate/sum it across all the
... training examples

% defining delta matrix which accumulates the value of gradient for each weight/parameter
... corresponding to Theta1 and Theta2 matrix
delta_prime_1 = zeros( size(Theta1,1) , size(Theta1,2) ) ;
delta_prime_2 = zeros( size(Theta2,1) , size(Theta2,2) ) ;

% doing operation using "for loop" on each individual data
for inp_dat= 1:m
    
    % layer 1 (input layer) computation for a single training example
    a_1 = [ 1 ; X(inp_dat,:)'] ; % we add 1 for the bias unit
    size(a_1,1);
    
    % layer 2 (hidden layer) computation for a single training example
    z_2 = Theta1 * a_1 ; 
    a_2 = [1; sigmoid(z_2)] ; 
    
    % layer 3 (output layer) computation for a single training example
    z_3= Theta2 * a_2 ; 
    a_3 = sigmoid(z_3) ; 
    h_x = a_3 ; ... its dimensions are kx1
    
    % backprop calculation at output layer (3rd layer)
    y_train_ex = y_vector(:,inp_dat) ; ... its dimensions are kx1
    delta_3 = a_3 - y_train_ex ; 
    
    % backprop calculation at hidden layer (2nd layer)
    
    Theta2_trunc = Theta2(:,2:end);
    delta_2 = ( (Theta2_trunc)' * delta_3 ) .* sigmoidGradient(z_2);
    
    % updating values of delta matrix which accumulates the value of gradient for
    .... each weight/parameter corresponding to Theta1 and Theta2 matrix
    delta_prime_1 = delta_prime_1 + ( delta_2 * (a_1)'  ) ; 
    delta_prime_2 = delta_prime_2 + ( delta_3 * (a_2)'  ) ; 
    
end

% finiding partial derivatives of cost function with respect to each weight/parameter
... we need to look at elements of the matrix to obtain the partial derivatives
... all these partial derivatives are then used to update the individual weights

Theta1_a = (1/m) * delta_prime_1 ; 
Theta2_a = (1/m) * delta_prime_2 ; 

% removing the column corresponding to the bias term
Theta1_b = Theta1_a(:,2:end) ;
Theta2_b = Theta2_a(:,2:end) ;

% regularizing all the non-bias terms.. we add the non-regularized terms with original parameters
... after multiplying with (lambda/m) term
Theta1_c = Theta1_b + (lambda/m)* Theta1(:,2:end);
Theta2_c = Theta2_b + (lambda/m)* Theta2(:,2:end);

% final regularized partial derivative terms
Theta1_grad = [Theta1_a(:,1) Theta1_c ] ;
Theta2_grad = [Theta2_a(:,1) Theta2_c ] ; 



%%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
