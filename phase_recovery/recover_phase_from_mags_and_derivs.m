clear all

ltfatstart(); % start the ltfat toolbox
phaseretstart;
%% Obtain data 
clipBelow = -10;

load('commands_listen.mat');

tfdata_amp = squeeze(generated(1:100,:,:,1));
tfdata_tgrad = squeeze(generated(1:100,:,:,2));
tfdata_fgrad = squeeze(generated(1:100,:,:,3));


%% STFT parameters 
% Set tolerance to be slightly above log-lower-limit
tol=[1e-1,1e-10];

a = 128;
M = 512;
tfr = 4;
win = {'gauss',tfr};
Ltrue = size(tfdata_amp,3)*a;
flag = 'freqinv';

win = gabwin(win,a,M,Ltrue);
dual = gabdual(win,a,M,Ltrue);
num_data = size(tfdata_amp, 1);
mask = zeros(M/2+1, size(tfdata_amp, 3));
usephase = zeros(M/2+1, size(tfdata_amp, 3));

%% Prepare arrays for results

to_listen = [];

%% Reconstruct signals
tic
for kk = 1:num_data    
    % Amplitude (original phase)
    c_amp = [squeeze(tfdata_amp(kk,:,:)); zeros(1,size(tfdata_amp,3))];
    c_amp = exp((-clipBelow/2)*(c_amp-1));   
    
    % Tgrad 
    tgrad = [squeeze(tfdata_tgrad(kk,:,:)); zeros(1,size(tfdata_amp,3))];
    tgrad = tgrad*250; % Unscale
    
    % Prescale
    tgrad = 2*pi*a/Ltrue * tgrad;     
    
    % Fgrad 
    fgrad = [squeeze(tfdata_fgrad(kk,:,:)); zeros(1,size(tfdata_amp,3))];
    fgrad = fgrad*750; % Unscale
    
    % Prescale   
    fgrad = - 2*pi/M * (fgrad + repmat((0:size(tfdata_amp,3)-1)*a,M/2+1,1)); 
    
    % Build phase
    [newphase, usedmask] = comp_constructphasereal(c_amp,tgrad,fgrad,a,M,tol,2,mask,usephase);

    % Build the coefficients
    c_amp_pghi = c_amp.*exp(1i*newphase);
    f_amp_pghi = idgtreal(c_amp_pghi,dual,a,M,flag);
    to_listen = [to_listen; f_amp_pghi; zeros(8000,1)];

    %audiowrite(strcat('audio_files/', num2str(kk), 'derivs_pghi.wav'), f_amp_pghi, 16000)
    if mod(kk,1000) == 0
        toc
        tic
        fprintf('-signal %d-',kk);
    end

end
toc

fs = 16000;
sound(to_listen, fs)