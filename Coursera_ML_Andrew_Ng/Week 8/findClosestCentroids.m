function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%finding number of examples in the dataset
m = size(X,1) ; 

%defining a matrix which would capture the norm(distance) of each example 
... from each centroid/cluster centre
norm_matrix = zeros(m,K)  ;

%updating idx variable for each example in the dataset
for i=1:m
    
    for j=1:K
        
        %finding norm of each example in the dataset
        norm_matrix(i,j) = norm( X(i,:) - centroids(j,:) ) ; 
        
    end
    
    %finding minimum norm from the K computed norms and also finding the
    ... index of the cluster which has minimum norm with a particular example
    [M,I] = min( norm_matrix(i,:) ,[], 2 ) ;
    
    %updating desired variable.. if there are multiple clusters with same norm..
    ... I will have multiple cluster values.. but we are interested in picking 
    ... only one of them
    idx(i,1) = I(1,1) ; 
    

end



% =============================================================

end
