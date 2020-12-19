function M = regmat(n,method)
%REGMAT  Generate a sparse regularization matrix.
%
%   M = REGMAT(N,METHOD,lambda) specifies the regularization method to use. Pass
%   in 'ridge' for METHOD to use ridge regression (default), 'Tikhonov' to
%   use Tikhonov regularization, or 'ols' to use ordinary least squares
%   (i.e., no regularization).
%
%   mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox

%   Authors: Mick Crosse <mickcrosse@gmail.com>
%            Edmund Lalor <edmundlalor@gmail.com>
%   Copyright 2014-2020 Lalor Lab, Trinity College Dublin.

% Set default values
if nargin < 2 || isempty(method)
    method = 'ridge';
end

% Generate a sparse matrix
switch method
    case 'ridge'
        M = sparse(eye(n));
        M(1,1) = 0;
    case 'Tikhonov'
        M = sparse(eye(n));
        M = M - 0.5*(diag(ones(n-1,1),1)+diag(ones(n-1,1),-1));
        M([n+2,end]) = 0.5;
        M([1,2,n+1]) = 0;
    case 'ols'
        M = sparse(zeros(n));
        
end