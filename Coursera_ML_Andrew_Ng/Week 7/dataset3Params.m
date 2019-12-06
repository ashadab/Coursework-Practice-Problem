function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%defining the set of constant C used in the SVM cost function
C_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

%defining the set of sigma used in the gaussian kernel
sigma_set = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
size_prime = size(C_set,1);

%initalizing the set capturing the error terms of the model for each
... possible combination of C and sigma
error_set = zeros(size_prime,size_prime);


%finding error for the SVM model.. where we train the model using training
... data and evaluate the model using validation data and compute the 
... error correspondignly over the validation set

for i = 1: size(C_set,1)
        
    for j = 1: size(sigma_set)
        
        %finding model parameters using training data
        model = svmTrain( X, y, C_set(i,1) , @(x1, x2) gaussianKernel(x1, x2, sigma_set(j,1) ) ) ;
        
        %finding predictions using validation set
        pred_prime = svmPredict(model, Xval);
        
        %finding error using how many predictions mismatches 
        error_set(i,j) = mean( double ( pred_prime ~= yval ) ) ; 
        
        fprintf('Value of error_set is %.4f when value of C is %.4f and value of sigma is %.4f', error_set(i,j), C_set(i,1),sigma_set(j,1)) ;  
        
    end
    
end

% finding values of C and sigma which gives minimum value of validation error
min_val = min( min( error_set ) ) ; 

[row_prime, col_prime] = find( error_set == min_val ) ; 

%in case we have multiple combinations which gives same minimum error.. 
...we would like to pick only the first value
row_final = row_prime(1,1);
col_final = col_prime(1,1);

%updating the final desired variables
C = C_set(row_final,1); 
sigma = sigma_set(col_final,1); 



% =========================================================================

end
