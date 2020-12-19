function lambdaset = makelambdaset(lambda, band)
k = numel(band);
base = numel(lambda);
if ~iscolumn(lambda), lambda = lambda'; end
% hinted from DEC2BASE:
x = (1:base^k)-1;
idx = zeros(base^k,k);
for power = (k-1):-1:0
  idx(:,k-power) = floor(x/base^power);
  x = rem(x,base^power);
end
lambdaset = lambda(idx+1);
end