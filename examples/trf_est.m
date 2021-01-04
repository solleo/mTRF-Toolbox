% Load data
load('data/speech_data.mat','stim','resp','fs');
stim = sum(stim,2);

% Normalize data
stim = zscore(stim);
resp = zscore(resp);

% lambdas = 10.^[5:20];
lambdas = 10.^[-5:5]; % it's too big anyway...
[Str,Rtr,Ste,Rte] = mTRFpartition(stim,resp,11,1);
stat_opt = mTRFcrossval(Str,Rtr,fs,1,-150,450,lambdas,'zeropad',0);
[~,idx] = max(mean(mean(stat_opt.r,1),3));

clf;
histogram(mean(stat_opt.err,3))
%%
mdl = mTRFtrain(stim,resp,fs,1,-50,450,1e3,'zeropad',0);

clf;
subplot(411)
imagesc(mdl.t, 1:128, squeeze(mdl.w(1,:,:))')
title('Weights')
subplot(412)
plot(mdl.t, mean(squeeze(mdl.w(1,:,:)-10),2))
title('mean')
% subplot(413)
% plot(mdl.t, std(squeeze(mdl.w(1,:,:)),[],2))
% title('std (ie, GFP)')
subplot(413)
plot(mdl.t, rms(squeeze(mdl.w(1,:,:)),2))
title('rms')
hold on; ylim0 = ylim;
line([109 109]',ylim0, 'color','r')
line([171 171]',ylim0, 'color','r')
line([351 351]',ylim0, 'color','r')

chanlocs = pop_readlocs('examples/biosemi128.ced');

subplot(4,3,10)
idx = find(mdl.t>109, 1,'first');
topoplot(squeeze(mdl.w(1,idx,:)), chanlocs);
caxis([-1 1]); colorbar

subplot(4,3,11)
idx = find(mdl.t>171, 1,'first');
topoplot(squeeze(mdl.w(1,idx,:)), chanlocs);
caxis([-1 1]); colorbar

subplot(4,3,12)
idx = find(mdl.t>351, 1,'first');
topoplot(squeeze(mdl.w(1,idx,:)), chanlocs);
caxis([-1 1]); colorbar

colormap(parula)
%%
mdl = mTRFtrain(stim,resp,fs,1,-50,450,1e3,'zeropad',1);
[pred,stat] = mTRFpredict(stim,resp,mdl, 'zeropad',1);
cla; hold on
plot(resp(:,1))
plot(pred(:,1))
sqrt(mean((resp(:,1)-pred(:,1)).^2))
%% PREDICTION SCALE IS MEANINGLESS WITH RIDGE SHRINKAGE...
L = 1e2;
mdl = mTRFtrain(stim,resp,fs,1,-50,450,L,'zeropad',1);
[pred,stat] = mTRFpredict(stim,resp,mdl, 'zeropad',1);
clf
subplot(211); hold on
plot((resp(:,1)))
plot((pred(:,1)))

subplot(212); hold on
plot(zscore(resp(:,1)))
plot(zscore(pred(:,1)))
