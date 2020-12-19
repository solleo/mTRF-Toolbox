function L = lambdamat(nvar,xvar,lambda,band)
%LAMBDAMAT generates banded or non-banded lambda matrix
%
%   L = lambdamat(nvar,xvar,lambda,arg.band)
%

%   Author: Seung-Goo Kim <solleo@gmail.com>

if isempty(band)  % default = [] when parsing inputs in MTRFTRAIN
  band = xvar;    % XVAR is determined by STIM(when dir=1)
end
nlags = (nvar-1)/xvar; % modeled lags: determined by ARG.TYPE
if ~isrow(lambda), lambda = lambda'; end
if ~isrow(band), band = band'; end
assert(numel(lambda)==numel(band),...
  '#lambda=%i =/= #bands=%i',numel(lambda),numel(band)) % Check # of bands
l = reshape(cell2mat(arrayfun(@(x,y) repmat(x,[nlags, y]), lambda, band, ...
  'UniformOutput',false))',[],1);  % diagonal elements
L = speye(nvar).*[0; l];  % first zero for offset
end