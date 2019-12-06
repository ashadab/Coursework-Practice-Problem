function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    
    %creating a variable giving prediction=1 if probability is less
    ... than epsilon, i.e. the example is an anomaly
    predictions = pval < epsilon ; 
    
    %creating true positive vector which returns 1 if condition is met
    tp = ( yval== 1 ) & ( predictions == 1 );
    
    %creating false negative vector which returns 1 if condition is met
    fn = ( yval== 1 ) & ( predictions == 0 );
    
    %creating false positive vector which returns 1 if condition is met
    fp = ( yval== 0 ) & ( predictions == 1 );
    
    % to find precision and recall number we need to sum of all the above 
    ... created vectors to determine total number of true +ve, false 
    ... -ve, false +ve
    
    precision = sum(tp)/ ( sum(tp) + sum(fp) ) ; 
    
    recall = sum(tp)/ ( sum(tp) + sum(fn) ) ; 
    
    F1 = ( 2 * precision * recall ) / ( precision+ recall ) ; 

    









    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
