import numpy as np
import heapq

__author__ = 'Andres'


def pghi(spectrogram, tgrad, fgrad, a, M, L, tol=10):
    """"Implementation of "A noniterativemethod for reconstruction of phase from STFT magnitude". by Prusa, Z., Balazs, P., and Sondergaard, P. Published in IEEE/ACM Transactions on Audio, Speech and LanguageProcessing, 25(5):1154–1164 on 2017. 
    a = hop size
    M = fft window size
    L = signal length
    tol = tolerance under the max value of the spectrogram
    """
    abstol = -20
    done_mask = np.zeros_like(spectrogram)
    phase = np.zeros_like(spectrogram)
    max_val = np.amax(spectrogram[done_mask == 0])
    max_pos = np.where(spectrogram==max_val)
       
    if max_val <= abstol:  #Avoid integrating the phase for the spectogram of a silent signal
        print('Empty spectrogram')
        return phase

    M2 = spectrogram.shape[0]
    N = spectrogram.shape[1]
    b =  L / M  
    
    sampToRadConst =  2.0 * np.pi / L # Rescale the derivs to rad with step 1 in both directions
    tgradw = a * tgrad * sampToRadConst
    fgradw = - b * ( fgrad + np.arange(spectrogram.shape[1]) * a ) * sampToRadConst # also convert relative to freqinv convention
                 
    magnitude_heap = []
    done_mask[spectrogram < max_val-tol] = 3 # Do not integrate over silence

    while np.any([done_mask==0]):
        max_val = np.amax(spectrogram[done_mask == 0]) # Find new maximum value to start integration
        max_pos = np.where(spectrogram==max_val)
        heapq.heappush(magnitude_heap, (-max_val, max_pos))
        done_mask[max_pos] = 1

        while len(magnitude_heap)>0: # Integrate around maximum value until reaching silence
            max_val, max_pos = heapq.heappop(magnitude_heap)
            
            col = max_pos[0]
            row = max_pos[1]
            
            #Spread to 4 direct neighbors
            N_pos = col+1, row
            S_pos = col-1, row
            E_pos = col, row+1
            W_pos = col, row-1

            if max_pos[0] < M2-1 and not done_mask[N_pos]:
                phase[N_pos] = phase[max_pos] + (fgradw[max_pos] + fgradw[N_pos])/2
                done_mask[N_pos] = 2
                heapq.heappush(magnitude_heap, (-spectrogram[N_pos], N_pos))

            if max_pos[0] > 0 and not done_mask[S_pos]:
                phase[S_pos] = phase[max_pos] - (fgradw[max_pos] + fgradw[S_pos])/2
                done_mask[S_pos] = 2
                heapq.heappush(magnitude_heap, (-spectrogram[S_pos], S_pos))

            if max_pos[1] < N-1 and not done_mask[E_pos]:
                phase[E_pos] = phase[max_pos] + (tgradw[max_pos] + tgradw[E_pos])/2
                done_mask[E_pos] = 2
                heapq.heappush(magnitude_heap, (-spectrogram[E_pos], E_pos))

            if max_pos[1] > 0 and not done_mask[W_pos]:
                phase[W_pos] = phase[max_pos] - (tgradw[max_pos] + tgradw[W_pos])/2
                done_mask[W_pos] = 2
                heapq.heappush(magnitude_heap, (-spectrogram[W_pos], W_pos))
    return phase
