function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %gives us number of input data
num_labels = size(Theta2, 1); %gives us the number of classes in which data will be classified

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% augmenting a column vector with 1s in X-input vector
X= [ ones(m,1) X ]; ... mXn .. where m:observations and n:#features


% a_2 gives us the activation value of nodes in 2nd-hidden layer
a_2 = sigmoid (X*Theta1') ; ... mX(n+1) * (n+1)X(nodes in layer2) =  mX(nodes in layer2)


% adding a column vector of "ones" in the 2nd-hidden layer
a_2 = [ ones(size(a_2,1), 1) a_2 ] ; ... mX(nodes in layer2 + 1)


% a_3 gives us the activation value of nodes in 3rd-output layer
a_3 = sigmoid (a_2*Theta2'); .... m X (nodes in layer2 + 1) * (nodes in layer2 + 1) X (num_labels) = m X num_labels

% in order to find predicted class we are capturing column index (I) of the maximum value (M) for each row  
[M I] = max(a_3,[],2); 

% in order to predict the class we are assigning prediction vector the index of the maximum value
p=I; 




% =========================================================================


end
