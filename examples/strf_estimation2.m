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
set(0,'DefaultTextFontSize',9)
load('../data/speech_data.mat','stim','resp','fs','factor');
chanlocs = pop_readlocs('biosemi128.ced');
stim = [stim, resample([zeros(1,16);diff(stim)]',1,4)'];

stim = resample(stim,1,2);
resp = resample(resp,1,2);

fs = 64;
band = [16 4];

% Z-scoring:
stim = zscore(stim);
resp = zscore(resp);

% EQUALIZING SUM of VARIANCE:
stim = [stim(:,1:band(1))/sum(std(stim(:,1:band(1)))), ...
  stim(:,1+band(1):sum(band))/sum(std(stim(:,1+band(1):sum(band))))];
fn_pdf = 'fig_ch106_16-4_zsc_eqsumvar.pdf';

%---------------------------------------------------
ax = axeslayout([4 6]);
axespos(ax,1)
t_stim = [0:size(stim,1)-1]/fs;
imagesc(t_stim, 1:16, stim(:,1:16)');
axis xy square; colorbaro
xlabel('Time [s]'); ylabel('Stim chan')
xlim([0 3])
title('Amplitude')

axespos(ax,7)
t_stim = [0:size(stim,1)-1]/fs;
imagesc(t_stim, 1:4, stim(:,17:20)');
axis xy square; h=colorbaro;
xlabel('Time [s]'); ylabel('Stim chan')
xlim([0 3])
title('Derivative')

axespos(ax,2)
imagesc(t_stim, 1:128, resp');
axis xy square; colorbaro
xlabel('Time [s]'); ylabel('EEG chan')
xlim([0 3])
title('Response')

% Set TRF parameters
lambdas = 10.^(-5:1:5);
ichan = 106;

% Estimate STRF model weights for amplitude only
nfolds = 10;
delete(gcp('nocreate'))
parpool(nfolds)

acc_eval1 = []; acc_opt = []; lambda_opt1=[];
parfor idx_te = 1:nfolds
  % partition: 10-fold
  [Str,Rtr,Ste,Rte] = mTRFpartition(stim(:,1:band(1)),resp(:,:),nfolds,idx_te);
  % optimize on 90%data (LOOCV)
  stat_opt = mTRFcrossval(Str,Rtr,fs,1,-100,400,lambdas,'zeropad',0);
  acc_opt(idx_te,:) = mean(mean(stat_opt.r,1),3);
  [~,idx] = max(mean(mean(stat_opt.r,1),3));
  lambda_opt1(idx_te) = lambdas(idx);
  % fit on 90%data
  model = mTRFtrain(Str,Rtr,fs,1,-100,400,lambda_opt1(idx_te),'zeropad',0);
  % evaluate on 10%holdout
  [~,stat_eval] = mTRFpredict(Ste,Rte,model,'zeropad',0);
  acc_eval1(idx_te,:) = stat_eval.r;
end

iaxes = 3;
mdlname = 'Amplitude-only';
[Str,Rtr,Ste,Rte] = mTRFpartition(...
  stim(:,1:band(1)),resp(:,:),nfolds,1);
showresults(ax,iaxes,lambdas,acc_eval1,acc_opt,lambda_opt1,...
  Str,Rtr,Ste,Rte,fs,mdlname,nfolds,chanlocs,ichan)

% Estimate STRF model weights for modulation phase only
acc_eval2 = []; acc_opt = []; lambda_opt2=[];
parfor idx_te = 1:nfolds
  % partition: 10-fold
  [Str,Rtr,Ste,Rte] = mTRFpartition(...
    stim(:,band(1)+1:sum(band)),resp(:,:),nfolds,idx_te);
  % optimize on 90%data (LOOCV)
  stat_opt = mTRFcrossval(Str,Rtr,fs,1,-100,400,lambdas,'zeropad',0);
  acc_opt(idx_te,:) = mean(mean(stat_opt.r,1),3); %#ok<*SAGROW>
  [~,idx] = max(mean(mean(stat_opt.r,1),3));
  lambda_opt2(idx_te) = lambdas(idx);
  % fit on 90%data
  model = mTRFtrain(Str,Rtr,fs,1,-100,400,lambda_opt2(idx_te),'zeropad',0);
  % evaluate on 10%holdout
  [~,stat_eval] = mTRFpredict(Ste,Rte,model,'zeropad',0);
  acc_eval2(idx_te,:) = stat_eval.r;
end

iaxes = 3+6;
mdlname = 'Deriv-only';
[Str,Rtr,Ste,Rte] = mTRFpartition(...
  stim(:,band(1)+1:sum(band)),resp(:,:),nfolds,1);
showresults(ax,iaxes,lambdas,acc_eval2,acc_opt,lambda_opt2,...
  Str,Rtr,Ste,Rte,fs,mdlname,nfolds,chanlocs,ichan)

% Joint model with a single lambda
acc_eval3 = []; acc_opt = []; lambda_opt3 = [];
parfor idx_te = 1:nfolds
  % partition: 10-fold
  [Str,Rtr,Ste,Rte] = mTRFpartition(stim,resp,nfolds,idx_te);
  % optimize on 90%data (LOOCV)
  stat_opt = mTRFcrossval(Str,Rtr,fs,1,-100,400,lambdas,'zeropad',0);
  acc_opt(idx_te,:) = mean(mean(stat_opt.r,1),3);
  [~,idx] = max(mean(mean(stat_opt.r,1),3));
  lambda_opt3(idx_te) = lambdas(idx);
  % fit on 90%data
  model = mTRFtrain(Str,Rtr,fs,1,-100,400,lambda_opt3(idx_te),'zeropad',0);
  % evaluate on 10%holdout
  [~,stat_eval] = mTRFpredict(Ste,Rte,model,'zeropad',0);
  acc_eval3(idx_te,:) = stat_eval.r;
end

iaxes = 3+6*2;
mdlname = 'Joint-1lambda';
[Str,Rtr,Ste,Rte] = mTRFpartition(stim,resp,nfolds,1);
showresults(ax,iaxes,lambdas,acc_eval3,acc_opt,lambda_opt3,...
  Str,Rtr,Ste,Rte,fs,mdlname,nfolds,chanlocs,ichan)

% Joint model with two lambdas
lambdaset = makelambdaset(lambdas, [16 4]);
acc_eval4 = []; acc_opt = []; lambda_opt4 = [];
parfor idx_te = 1:nfolds
  % partition: 10-fold
  [Str,Rtr,Ste,Rte] = mTRFpartition(stim,resp,nfolds,idx_te);
  % optimize on 90%data (LOOCV)
  stat_opt = mTRFcrossval(Str,Rtr,fs,1,-100,400,lambdas,...
    'band',band,'zeropad',0);
  acc_opt(idx_te,:) = mean(mean(stat_opt.r,1),3);
  [~,idx] = max(mean(mean(stat_opt.r,1),3));
  lambda_opt4(idx_te,:) = lambdaset(idx,:);
  % fit on 90%data
  model = mTRFtrain(Str,Rtr,fs,1,-100,400,lambda_opt4(idx_te,:),...
    'band',band,'zeropad',0);
  % evaluate on 10%holdout
  [~,stat_eval] = mTRFpredict(Ste,Rte,model,'zeropad',0);
  acc_eval4(idx_te,:) = stat_eval.r;
end
delete(gcp('nocreate'))

iaxes = 3+6*3;
mdlname = 'Joint-2lambdas';
[Str,Rtr,Ste,Rte] = mTRFpartition(stim,resp,nfolds,1);
showresults(ax,iaxes,lambdas,acc_eval4,acc_opt,lambda_opt4,...
  Str,Rtr,Ste,Rte,fs,mdlname,nfolds,chanlocs,ichan)

% No further optimization.
axespos(ax,19)
cat_plot_boxplot_sg([acc_eval1(:,ichan), acc_eval2(:,ichan),...
  acc_eval3(:,ichan),acc_eval4(:,ichan)],...
  struct('showdata',3,'groupcolor',brewermap(4,'Set2')))
set(gca,'xtick',1:4,'xticklabel',{'Amp','Deriv','Joint1','Joint2'})
title({sprintf('10%%holdout x %i',nfolds),['ch',num2str(ichan)]})

h = gcf;
h.Position = [39         211        1230         735];
h.PaperOrientation='landscape';
print(fn_pdf,'-dpdf','-bestfit')
unix(['pdfcrop ',fn_pdf,' ',fn_pdf])

%%
function showresults(ax,iaxes,lambdas,acc_eval,acc_opt,lambda_opt,Str,Rtr,Ste,Rte,fs,mdlname,nfolds,chanlocs,ichan)
axespos(ax,iaxes)
hold on
if isvector(lambda_opt)
  h = errorplot(lambdas, acc_opt);
  h.patch.FaceAlpha=0.5;
  ylim0 = ylim;
  line(repmat(lambda_opt,[2 1]), repmat([-1 1]',[1 nfolds]),'color','r')
  [fi,xi] = ksdensity(log10(lambda_opt));
  plot(10.^xi,ylim0(1)+fi/max(fi)*(ylim0(2)-ylim0(1))*0.2,'color',[1 .5 0])
  lambda_med = median(lambda_opt);
  line(repmat(lambda_med,[2 1]), [-1 1]','color',[0 .7 0])
  set(gca,'xscale','log'); ylabel('Acc\pmSE');xlabel('\lambda')
  ylim(ylim0); xlim([lambdas(1) lambdas(end)])
else
  nL = numel(lambdas);
  imagesc(log10(lambdas),log10(lambdas),reshape(mean(acc_opt,1),[nL,nL]))
  xlabel('log_{10}\lambda(f1)'); ylabel('log_{10}\lambda(f2)')
  h=colorbaro; title(h,'Acc')
  axis square xy
  scatter(log10(lambda_opt(:,1))+(rand(nfolds,1)-0.5)/5, ...
    log10(lambda_opt(:,2))+(rand(nfolds,1)-0.5)/5, 'c.')
  colormap(gca,brewermap(128,'Reds'))
  lambda_med = median(lambda_opt);
  scatter(log10(lambda_med(1)),log10(lambda_med(2)),'g*')
end
title({mdlname,'Inner-LOOCV:90%data'})

axespos(ax,iaxes+1)
if isempty(ichan)
  [~,ichan] = max(mean(acc_eval,1));
end
topoplot(mean(acc_eval,1), chanlocs,'electrodes','on',...
  'emarker2',{ichan,'.',[0 .7 0]})
caxis([0 .1])
h = colorbaro;
title(h,'Acc')
colormap(gca,brewermap(128,'Reds'))
title({sprintf('Eval 10%%holdout x %i', nfolds),...
  sprintf('r[ch%i]=%.2f', ichan, median(acc_eval(:,ichan),1))})

axespos(ax,iaxes+2)
if isvector(lambda_opt)
  model = mTRFtrain(Str,Rtr,fs,1,-100,400,lambda_med,'zeropad',0);
  imagesc(model.t, 1:size(model.w,1), model.w(:,:,ichan))
  title({sprintf('TRF[ch%i]',ichan),sprintf('L=%1.2g',lambda_med)})
else
  model = mTRFtrain(Str,Rtr,fs,1,-100,400,lambda_med,...
    'zeropad',0 ,'band',[16 4]);
  imagesc(model.t, 1:size(model.w,1), model.w(:,:,ichan))
  title({sprintf('TRF[ch%i]',ichan),sprintf('L=%1.2g;%1.2g',lambda_med)})
end
xlabel('Lag [ms]'); ylabel('Features')
xlim([-50 350]); axis xy square; h = colorbaro; title(h,'w')

axespos(ax,iaxes+3)
[pred,stats] = mTRFpredict(Ste,Rte,model,'zeropad',1);
hold on
plot((0:size(Rte,1)-1)/fs, zscore(Rte(:,ichan)),'k')
plot((0:size(pred,1)-1)/fs, zscore(pred(:,ichan)),'r')
title({sprintf('Pred[%ithfold]',1),...
  sprintf('ch%i:r=%.2f',ichan, stats.r(ichan))})
xlabel('Time [s]'); xlim([0 3])
ylabel('Z-score')
axis square

drawnow
end