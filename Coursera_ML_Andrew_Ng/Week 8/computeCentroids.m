function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.

centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%



% creating a matrix where each row corresponds to coordinates of a point
... and the corresponding cluster assigned to that data point
comb = [X idx]; 

% running the loop for K centroids
for j=1:K
    
    % finding out data points (rows) which are assigned to cluster "j"
    ty = comb( comb(:, n+1) == j, : ) ; 
    
    %finding mean of each of the n-coordinates for the data points (rows of "comb" matrix) 
    ... which are assigned to a particular centroid
    centroids(j,:) = mean( ty (:, 1 : end-1 ) , 1 ) ; 
    

end

    
    


% =============================================================


end

