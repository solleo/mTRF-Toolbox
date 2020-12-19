% ### TRF/STRF estimation
%
% Here, we estimate a 16-channel spectro-temporal response function (STRF)
% from 2 minutes of EEG data recorded while a human participant listened to
% natural speech. To map in the forward direction (encoding model), we set
% the direction of causality to 1. To capture the entire STRF timecourse,
% the time lags are computed between -100 and 400 ms. The regularization
% parameter is set to 0.1 to reduce overfitting to noise.
tab = readtable('/home/sgk/Downloads/biosemi128.csv');
writetable(tab,'biosemi128.ced','FileType','Text','Delimiter','tab')

%% Load example speech dataset

clear; clf
load('../data/speech_data.mat','stim','resp','fs','factor');
chanlocs = pop_readlocs('biosemi128.ced');
stim = [stim, angle(hilbert(stim))];

subplot(551)
t_stim = [0:size(stim,1)-1]/fs;
imagesc(t_stim, 1:16, stim(:,1:16)');
axis xy square; colorbaro
xlabel('Time [s]'); ylabel('Stim chan')
xlim([0 3])
title('Amplitude')

subplot(556)
t_stim = [0:size(stim,1)-1]/fs;
imagesc(t_stim, 1:16, stim(:,17:32)');
axis xy square; colorbaro
xlabel('Time [s]'); ylabel('Stim chan')
xlim([0 3])
title('Amplitude modulation phase')
hmap(1:256,1) = linspace(0,1,256);
hmap(:,[2 3]) = 0.7; % saturation & brightness
huemap = hsv2rgb(hmap);
colormap(gca,huemap)

subplot(552)
imagesc(t_stim, 1:128, resp');
axis xy square; colorbaro
xlabel('Time [s]'); ylabel('EEG chan')
xlim([0 3])

% Set TRF parameters
% ichan = 1:128; % 85=Fz, 87=FCz, 1=Cz
lambdas = 2.^(-10:2:10);

% Estimate STRF model weights for amplitude only
nfolds = 10;
r = []; meanr = []; lambda_opt1=[];
for idx_te = 1:nfolds
  % optimize
  [S,R,Ste,Rte] = mTRFpartition(...
    stim(:,1:16),resp(:,:),nfolds,idx_te);
  stat = mTRFcrossval(S,R,fs,1,-100,400,lambdas,'zeropad',0);
  [~,idx] = max(mean(mean(stat.r,1),3));
  lambda_opt1(idx_te) = lambdas(idx);
  % fit
  model = mTRFtrain(S,R,fs,1,-100,400,lambda_opt1(idx_te),'zeropad',0);
  % evaluate
  [~,stats] = mTRFpredict(Ste,Rte,model,'zeropad',0);
  r(idx_te,:) = stats.r;
  meanr(idx_te,:) = mean(mean(stat.r,1),3);
end

subplot(553); hold on
plot(lambdas, mean(meanr),'.-'); ylim0 = ylim;
plot([mean(lambda_opt1) mean(lambda_opt1)], [-1 1]);
text(mean(lambda_opt1),ylim0(1),sprintf('%.2f',lambdas(idx)))
set(gca,'xscale','log'); ylim(ylim0); ylabel('Acc');xlabel('\lambda')
title({'Amplitude-only','Inner-LOOCV'})

subplot(554)
topoplot(mean(r,1), chanlocs,'electrodes','on')
caxis([0 0.1])
colorbaro
colormap(gca,brewermap(128,'Reds'))
[~,ichan] = max(mean(r,1));

subplot(555)
model = mTRFtrain(S,R,fs,1,-100,400,mean(lambda_opt1),'zeropad',0);
imagesc(model.t, 1:16, model.w(:,:,ichan))
title({sprintf('mTRF:chan%i:L=%.2g',ichan,mean(lambda_opt1))})
xlabel('Lag [ms]')
xlim([-50 350]); axis xy square; colorbaro

%%
subplot(555)
[pred,stats] = mTRFpredict(Ste,Rte,model,'zeropad',0);
hold on
plot(zscore(Rte),'k')
plot(zscore(pred),'r')
title(sprintf('Pred:chan%i:r=%.2f',ichan, stats.r))
% xlabel('Time [s]'); 
ylabel('Z-score')
axis square

%% Estimate STRF model weights for modulation phase only
[S,R,Ste,Rte] = mTRFpartition(stim(:,17:32),resp(:,ichan),nfolds,idx_te);
stat = mTRFcrossval(S,R,fs,1,-100,400,lambdas);
[~,idx] = max(mean(stat.r,1));
lambda_opt2 = lambdas(idx);

subplot(5,5,8)
hold on
plot(lambdas, mean(stat.r,1),'.-')
ylim0 = ylim;
plot([lambdas(idx) lambdas(idx)], [-1 1]);
set(gca,'xscale','log')
ylim(ylim0)
title({'Amplitude-only','Inner-LOOCV'})
ylabel('Acc');xlabel('\lambda')

model = mTRFtrain(S,R,fs,1,-100,400,lambda_opt2);  

subplot(5,5,9)
imagesc(model.t, 1:16, zscore(model.w(:,:,1)')',[-3 3])
title({'Phase-only',sprintf('mTRF:chan%i:L=%.2g',ichan,lambda_opt2)})
xlabel('Lag [ms]')
xlim([-50 350]); axis xy square; colorbaro

subplot(5,5,10)
[pred,stats] = mTRFpredict(Ste,Rte,model);
hold on
plot(zscore(Rte),'k')
plot(zscore(pred),'r')
title(sprintf('Pred:chan%i:r=%.4f',ichan, stats.r))
xlabel('resp'); ylabel('pred')
axis square

%% Joint model with a single lambda
[S,R] = mTRFpartition(...
  stim(idx_tr,:),resp(idx_tr,ichan)*factor,10);
stat = mTRFcrossval(S,R,fs,1,-100,400,lambdas);
[~,idx] = max(mean(stat.r,1));
lambda_opt3 = lambdas(idx);
model = mTRFtrain(stim(idx_tr,:),resp(idx_tr,ichan)*factor,fs,1,-100,400,...
  lambda_opt3);

subplot(4,4,11)
imagesc(model.t, 1:32, zscore(model.w(:,:,1)')')
title({'Joint-1lambda',sprintf('mTRF:chan%i:L=%.2g',ichan,lambda_opt3)})
xlabel('Lag [ms]')
axis xy square; colorbaro

subplot(4,4,12)
[pred,stats] = mTRFpredict(stim(idx_te,:),resp(idx_te,ichan)*factor,model);
hold on;
plot(t_stim(idx_te), zscore(resp(idx_te,ichan)),'k')
plot(t_stim(idx_te), zscore(pred(:,1)),'r')
xlim([100 105])
title(sprintf('Pred:chan%i:r=%.4f',ichan, stats.r(1)))
xlabel('resp'); ylabel('pred')
axis square

% Joint model with two lambdas
[S,R] = mTRFpartition(...
  stim(idx_tr,:),resp(idx_tr,ichan)*factor,10);
% need to do joint-optimization (takes ^#BANDS time!)
stat = mTRFcrossval(S,R,fs,1,-100,400,lambdas,'band',[16 16]);

[~,idx] = max(mean(stat.r,1));
lambda_opt4 = stat.lambdaset(idx,:);

model = mTRFtrain(stim(idx_tr,:),resp(idx_tr,ichan)*factor,...
  fs,1,-100,400,lambda_opt4,'band',[16 16]);

subplot(4,4,15)
imagesc(model.t, 1:32, zscore(model.w(:,:,1)')',[-3 3])
title({'Joint-2lambdas',...
  sprintf('mTRF:chan%i:L1=%.2g:L2=%.2g',ichan,lambda_opt4)})
xlabel('Lag [ms]')
axis xy square; colorbaro

subplot(4,4,16)
[pred,stats] = mTRFpredict(stim(idx_te,:),resp(idx_te,ichan)*factor,model);
hold on;
plot(t_stim(idx_te), zscore(resp(idx_te,ichan)),'k')
plot(t_stim(idx_te), zscore(pred(:,1)),'r')
xlim([100 105])

title(sprintf('Pred:chan%i:r=%.4f',ichan, stats.r(1)))
xlabel('resp'); ylabel('pred')
axis square


%% No further optimization.
% cv = cvpartition(size(stim,1),'holdout',0.2);
r_amp = []; r_phase = []; r_unbanded = []; r_banded = [];
for k = 1:10
%   cv = repartition(cv);
%   idx_te = test(cv);
%   idx_tr = training(cv);
  
  model = mTRFtrain(stim(idx_tr,1:16),resp(idx_tr,ichan)*factor,...
    fs,1,-100,400,lambda_opt1,'verbose',0);
  [~,stats] = mTRFpredict(stim(idx_te,1:16),resp(idx_te,ichan)*factor,...
    model,'verbose',0);
  r_amp = [r_amp stats.r(1)];
  
  model = mTRFtrain(stim(idx_tr,17:32),resp(idx_tr,ichan)*factor,...
    fs,1,-100,400,lambda_opt2,'verbose',0);
  [~,stats] = mTRFpredict(stim(idx_te,17:32),resp(idx_te,ichan)*factor,...
    model,'verbose',0);
  r_phase = [r_phase stats.r(1)];
  
  model = mTRFtrain(stim(idx_tr,:),resp(idx_tr,ichan)*factor,...
    fs,1,-100,400,lambda_opt3,'verbose',0);
  [~,stats] = mTRFpredict(stim(idx_te,:),resp(idx_te,ichan)*factor,...
    model,'verbose',0);
  r_unbanded = [r_unbanded stats.r(1)];
  
  model = mTRFtrain(stim(idx_tr,:),resp(idx_tr,ichan)*factor,...
    fs,1,-100,400,lambda_opt4,'band',[16 16],'verbose',0);
  [~,stats] = mTRFpredict(stim(idx_te,:),resp(idx_te,ichan)*factor,...
    model,'verbose',0);
  r_banded = [r_banded stats.r(1)];
end
%
subplot(4,4,13)
cat_plot_boxplot_sg([r_amp; r_phase; r_unbanded; r_banded],...
  struct('showdata',3,'groupcolor',brewermap(4,'Set2')))
set(gca,'xtick',1:4,'xticklabel',{'Amp','Phase','Joint1','Joint2'})
title('20 rep 20%-holdout')

%% We compute the broadband TRF by averaging the STRF model across frequency
% channels and the global field power (GFP) by taking the standard
% deviation across EEG channels, and plot them as a function of time lags.
% This example can also be generated using
% [plot_speech_STRF](examples/plot_speech_strf.m) and
% [plot_speech_TRF](examples/plot_speech_trf.m).

% % Plot STRF
% figure
% subplot(2,2,1), mTRFplot(model,'mtrf','all',85,[-50,350]);
% title('Speech STRF (Fz)'), ylabel('Frequency band'), xlabel('')
%
% % Plot GFP
% subplot(2,2,2), mTRFplot(model,'mgfp','all','all',[-50,350]);
% title('Global Field Power'), xlabel('')
%
% % Plot TRF
% subplot(2,2,3), mTRFplot(model,'trf','all',85,[-50,350]);
% title('Speech TRF (Fz)'), ylabel('Amplitude (a.u.)')
%
% % Plot GFP
% subplot(2,2,4), mTRFplot(model,'gfp','all','all',[-50,350]);
% title('Global Field Power')
