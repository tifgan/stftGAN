# Module to download the dataset.

import librosa
from data.strechableNumpyArray import StrechableNumpyArray
import numpy as np
import os
from data.ourLTFATStft import LTFATStft
import ltfatpy
from data.modGabPhaseGrad import modgabphasegrad
ltfatpy.gabphasegrad = modgabphasegrad # This function is not implemented for one sided stfts with the phase method on ltfatpy
import scipy.io

def load_data(pathToBaseDatasetFolder, folderNames):
    dirs = [pathToBaseDatasetFolder+folderName for folderName in folderNames]
    audios = StrechableNumpyArray()
    i = 0
    total = 0
    for directory in dirs:
        print(directory)
        for file_name in os.listdir(directory):
            if file_name.endswith('.wav'):      
                audio, sr = librosa.load(directory + '/' + file_name, sr=None, dtype=np.float64)

                if len(audio) < 16000:
                    before = int(np.floor((16000-len(audio))/2))
                    after = int(np.ceil((16000-len(audio))/2))
                    audio = np.pad(audio, (before, after), 'constant', constant_values=(0, 0))
                if np.sum(np.absolute(audio)) < len(audio)*1e-4: 
                    continue

                audios.append(audio[:16000])
                i+=1

                if i > 1000:
                    i -= 1000
                    total += 1000
                    print("Just loaded 1000 files! The total now is:", total)
    print("Finished! I loaded", total+i, "audio files.")

    audios = audios.finalize()
    audios = np.reshape(audios, (total+i, len(audio))).astype(np.float64)
    print("audios shape:", audios.shape)
    return audios

clipBelow = -10

def generate_spectrograms_and_derivs_from(audio_signals):
    fft_hop_size = 128
    fft_window_length = 512
    L = 16384

    anStftWrapper = LTFATStft()
    spectrograms = np.zeros([len(audio_signals), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
    tgrads = np.zeros([len(audio_signals), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
    fgrads = np.zeros([len(audio_signals), int(fft_window_length//2+1), int(L/fft_hop_size)], dtype=np.float64)
    print("spectrograms shape:", spectrograms.shape)
    gs = {'name': 'gauss', 'M': 512}
        
    for index, audio_signal in enumerate(audio_signals):
        realDGT = anStftWrapper.oneSidedStft(signal=audio_signal, windowLength=fft_window_length, hopSize=fft_hop_size)
        spectrogram = anStftWrapper.logMagFromRealDGT(realDGT, clipBelow=np.e**clipBelow, normalize=True)
        spectrograms[index] = spectrogram  
        tgradreal, fgradreal = ltfatpy.gabphasegrad('phase', np.angle(realDGT), fft_hop_size,
                                                    fft_window_length)
        tgrads[index] = tgradreal /64
        fgrads[index] = fgradreal /256
    return spectrograms, tgrads, fgrads

def save_matrices(spectrograms, tgrads, fgrads):
    nameForFile = 'data/test_spectrograms_and_derivs'

    shiftedSpectrograms = spectrograms/(-clipBelow/2)+1
    countPerFile = 4000 # mat files sadly cannot be arbitrarily large. 4000 works for 3 matrices (mag+tderiv+fderiv).

    for index in range(1 + len(spectrograms)//countPerFile):
        scipy.io.savemat(nameForFile + '_' + str(index+1) + '.mat', dict(logspecs=shiftedSpectrograms[index*countPerFile:(index+1)*countPerFile], 
                                                                   tgrad=tgrads[index*countPerFile:(index+1)*countPerFile], 
                                                                   fgrad=fgrads[index*countPerFile:(index+1)*countPerFile]))

    
if __name__ == '__main__':
    pathToBaseDatasetFolder = 'data/sc09/'
    folderNames = ['train', 'test', 'valid']

    print('start loading the data')
    audio_signals = load_data(pathToBaseDatasetFolder, folderNames)

    print('compute spectrograms and derivs')
    spectrograms, tgrads, fgrads = generate_spectrograms_and_derivs_from(audio_signals)
    
    print('save everything')
    save_matrices(spectrograms, tgrads, fgrads)
    

