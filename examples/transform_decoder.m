function transform_decoder
%TRANSFORM_DECODER  Transform example decoder into forward encoding model.
%   TRANSFORM_DECODER loads an example dataset, trains a neural decoder and
%   transforms it into a neurophysiologically interpretable forward
%   encoding model (TRF) from 2 minutes of 128-channel EEG data as per
%   Haufe et al. (2014).
%
%   Example data is loaded from SPEECH_DATA.MAT and consists of the
%   following variables:
%       'stim'      a vector containing the speech spectrogram, obtained by
%                   band-pass filtering the speech signal into 128
%                   logarithmically spaced frequency bands between 100
%                   and 4000Hz, taking the Hilbert transform at each band
%                   and averaging over every 8 neighbouring bands.
%       'resp'      a matrix containing 2 minutes of 128-channel EEG data
%                   filtered between 0.5 and 15 Hz
%       'fs'        the sample rate of STIM and RESP (128Hz)
%       'factor'    the BioSemi EEG normalization factor for converting the
%                   TRF to microvolts (524.288mV / 2^24bits)
%
%   mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox

%   References:
%      [1] Haufe S, Meinecke F, Gorgen K, Dahne S, Haynes JD, Blankertz B,
%          Bie�mann F (2014) On the interpretation of weight vectors of
%          linear models in multivariate neuroimaging. NeuroImage
%          87:96-110.

%   Authors: Mick Crosse <mickcrosse@gmail.com>
%   Copyright 2014-2020 Lalor Lab, Trinity College Dublin.

% Load data
load('data/speech_data.mat','stim','resp','fs');

% Normalize data
stim = sum(stim,2);
resp = resp/std(resp(:));

% Downsample data
fsNew = 64;
stim = resample(stim,fsNew,fs);
resp = resample(resp,fsNew,fs);
fs = fsNew;

% Model parameters
dir = -1;
tmin = -150;
tmax = 450;
lambda = 256;

% Compute backward model weights
bmodel = mTRFtrain(stim,resp,fs,dir,tmin,tmax,lambda,'zeropad',0);

% Transform to forward model weights
fmodel = mTRFtransform(bmodel,resp,'zeropad',0);

% Define ROI
chan = 85; % channel Fz

% Plot TRF
figure, subplot(1,2,1)
plot(fmodel.t,squeeze(fmodel.w(chan,:)),'linewidth',3);
xlim([-50,350]);
title('Frontal TRF (Fz)')
xlabel('Time lag (ms)')
ylabel('Amplitude (a.u.)')
axis square, grid on

% Plot GFP
subplot(1,2,2)
area(fmodel.t,squeeze(std(fmodel.w)),'edgecolor','none');
xlim([-50,350])
title('Global Field Power')
xlabel('Time lag (ms)')
axis square, grid on