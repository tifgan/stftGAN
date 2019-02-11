%clear all

ltfatstart(); 
phaseretstart;
%% Obtain data 
clipBelow = -10;

load('commands_listen.mat');
tfdata_amp = exp((-clipBelow/2)*(generated(1:100, :, :)-1));   
tfdata_amp = [tfdata_amp zeros(size(tfdata_amp, 1), 1, size(tfdata_amp,3))]; % Add nyquist frequency
num_data = size(tfdata_amp, 1);


%% STFT parameters 

a = 128;
M = 512;
tfr = 4;
win = {'gauss', tfr};
flag = 'freqinv';
Ltrue = size(tfdata_amp,3)*a;

win = gabwin(win,a,M,Ltrue);
dual = gabdual(win,a,M,Ltrue);
gamma = pghi_findgamma(win,a,M,Ltrue);
mask = zeros(M/2+1, size(tfdata_amp, 3));

%% Prepare arrays for results

to_listen = [];

%% Reconstruct signals
tic
for kk = 1:num_data    
    % Amplitude (original phase)
    c_amp = squeeze(tfdata_amp(kk,:,:)); 
    c_amp_pghi = pghi(c_amp,gamma,a,M,mask,flag);
    f_amp_pgla = idgtreal(c_amp_pghi,dual,a,M,flag);
    
    to_listen = [to_listen; f_amp_pgla; zeros(8000,1)];
    %audiowrite(strcat('audio_files/', num2str(kk), '.wav'), f_amp_pgla,
    %16000) % uncomment to write audio files

    if mod(kk,1000) == 0
        toc
        tic
        fprintf('-signal %d-',kk);
    end

end
toc

fs = 16000;

sound(to_listen, fs)