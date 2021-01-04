clear
rng(1234)
X = randn(100,5);
X(:,2) = X(:,3)+X(:,4);
X(:,4) = X(:,1)+X(:,5);

b = [1 2 -3 4 5]';
Y = X*b + randn(100,1)*20;
X = zscore(X);
Y = zscore(Y);

Xtr = X(1:50,:);    Ytr = Y(1:50);
Xte = X(51:100,:);  Yte = Y(51:100);

lambdas = [0 2.^[-5:20]];
bhats = []; r =[];
GCVE = []; MSE = [];
for i = 1:numel(lambdas)
  L = lambdas(i);
  A = (Xtr'*Xtr+L*eye(5)) \ Xtr';
  bhat = A*Ytr;
  bhats = [bhats bhat];
  Yhat = Xte*bhat;
  MSE(i) = mean( (Yte-Yhat).^2 );
  r(i) = corr(Yte, Yhat);
%   GCVE(i) = MSE(i)/(size(Yte,1)-trace( Xtr*((Xtr'*Xtr+L*eye(5))\Xtr') ));
  N = size(Yte,1);
  S = Xtr * A;
  GCVE(i) = mean( ( (Yte-Yhat)./(1-trace(S)/N) ).^2 ); % Hastie Eq (7.52)
end
clf
subplot(421)
plot(lambdas, bhats, '.-')
set(gca,'xscale','log')

subplot(423); hold on
plot(lambdas, MSE, '.-')
plot(lambdas, lambdas*0+mean(Yte.^2))
[~,idx] = min(MSE);
scatter(lambdas(idx), MSE(idx), 'r')
ylabel('MSE'); set(gca,'xscale','log')

subplot(425); hold on
plot(lambdas, GCVE, '.-')
[~,idx] = min(GCVE);
scatter(lambdas(idx), GCVE(idx), 'r')
ylabel('GCVE'); set(gca,'xscale','log')


subplot(427); hold on
plot(lambdas, r, '.-')
[~,idx] = max(r);
scatter(lambdas(idx), r(idx), 'r')
ylabel('Pearson corr'); set(gca,'xscale','log')

subplot(424); hold on
plot(Yte)
[~,idx] = min(MSE);
Yhat = Xte*bhats(:,idx);
plot(Yhat)
title(['lambda = ',num2str(lambdas(idx)),'; R^2=',num2str(r2vec(Yte,Yhat))])

subplot(426); hold on
plot((Yte))
[~,idx] = min(GCVE);
Yhat = Xte*bhats(:,idx);
plot(Yhat)
title(['lambda = ',num2str(lambdas(idx)),'; R^2=',num2str(r2vec(Yte,Yhat))])

subplot(428); hold on
plot(zscore(Yte))
[~,idx] = max(r);
Yhat = Xte*bhats(:,idx);
plot(Yhat)
title(['lambda = ',num2str(lambdas(idx)),'; R^2=',num2str(r2vec(Yte,Yhat))])





% so THIS LARGE lambda is SERIOUSLY UNDERFITTING...
% Pearson correlation WILL lead to SERIOUS UNDERFITTING when lambda is
% large (because corr(Y,eps) = 1)

