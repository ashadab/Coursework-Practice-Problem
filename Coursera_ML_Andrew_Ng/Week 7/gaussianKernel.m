function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%


% we should do feature scaling to find similarity function
... but not doing it since it is not required for this data..
... since x1 and x2 are in similar range
% x1_scal = (x1-mean(x1))/ std(x1) ;
% x2_scal = (x2-mean(x2))/ std(x2) ;


var_sig = sigma*sigma;

sim = exp( - ( 1/(2*var_sig) ) * ( norm( x1- x2 ) )^2  );


% =============================================================
    
end
