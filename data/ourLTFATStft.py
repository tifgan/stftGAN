import ltfatpy
import numpy as np

__author__ = 'Andres'


class LTFATStft(object):
    def oneSidedStft(self, signal, windowLength, hopSize):
        gs = {'name': 'gauss', 'M': windowLength}
        return ltfatpy.dgtreal(signal, gs, hopSize, windowLength)[0]

    def inverseOneSidedStft(self, signal, windowLength, hopSize):
        synthesis_window = {'name': 'gauss', 'M': windowLength}
        analysis_window = {'name': ('dual', synthesis_window['name']), 'M': synthesis_window['M']}

        return ltfatpy.idgtreal(signal, analysis_window, hopSize, windowLength)[0]

    def magAndPhaseOneSidedStft(self, signal, windowLength, hopSize):
        stft = self.oneSidedStft(signal, windowLength, hopSize)
        return np.abs(stft), np.angle(stft)

    def log10MagAndPhaseOneSidedStft(self, signal, windowLength, hopSize, clipBelow=1e-14):
        realDGT = self.oneSidedStft(signal, windowLength, hopSize)
        return self.log10MagFromRealDGT(realDGT, clipBelow), np.angle(realDGT)

    def log10MagFromRealDGT(self, realDGT, clipBelow=1e-14):
        return np.log10(np.clip(np.abs(realDGT), a_min=clipBelow, a_max=None))

    def reconstructSignalFromLogged10Spectogram(self, logSpectrogram, phase, windowLength, hopSize):
        reComplexStft = (10 ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft, windowLength, hopSize)

    def logMagAndPhaseOneSidedStft(self, signal, windowLength, hopSize, clipBelow=np.e**-30, normalize=False):
        realDGT = self.oneSidedStft(signal, windowLength, hopSize)
        spectrogram = self.logMagFromRealDGT(realDGT, clipBelow, normalize)
        return spectrogram, np.angle(realDGT)

    def logMagFromRealDGT(self, realDGT, clipBelow=np.e**-30, normalize=False):
        spectrogram = np.abs(realDGT)
        if normalize:
            spectrogram = spectrogram/np.max(spectrogram)
        return np.log(np.clip(spectrogram, a_min=clipBelow, a_max=None))

    def reconstructSignalFromLoggedSpectogram(self, logSpectrogram, phase, windowLength, hopSize):
        reComplexStft = (np.e ** logSpectrogram) * np.exp(1.0j * phase)
        return self.inverseOneSidedStft(reComplexStft, windowLength, hopSize)
