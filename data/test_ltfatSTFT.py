from unittest import TestCase

import numpy as np

from ourLTFATStft import LTFATStft

__author__ = 'Andres'


class TestStftWrapper(TestCase):
    @classmethod
    def setUpClass(self):
        self.signal_length = 5120
        self.sampling_rate = 44100
        self.fft_window_length = 512
        self.fft_hop_size = 128
        self.fake_time = np.arange(0, self.signal_length / self.sampling_rate, 1 / self.sampling_rate, dtype=np.float64)
        self.fake_signal = np.sin(2 * np.pi * 440 * self.fake_time, dtype=np.float64)
        self.fake_noise = np.random.normal(0, 1, size=self.signal_length)

    def setUp(self):
        self.anStftWrapper = LTFATStft()

    def test01TheOneSidedStftHasTheRightAmountOfFrequencyBins(self):
        def test01(signal):
            oneSidedStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                           hopSize=self.fft_hop_size)
            self.assertEquals(oneSidedStft.shape[0], self.fft_window_length // 2 + 1)
        test01(self.fake_signal)
        test01(self.fake_noise)

    def test02TheOneSidedStftHasTheRightAmountOfTimeSteps(self):
        def test02(signal):
            oneSidedStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                           hopSize=self.fft_hop_size)
            self.assertEquals(oneSidedStft.shape[1], self.signal_length // self.fft_hop_size)
        test02(self.fake_signal)
        test02(self.fake_noise)

    def test03TheInverseOneSidedStftHasTheRightLenght(self):
        def test03(signal):
            oneSidedStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                           hopSize=self.fft_hop_size)
            timeSignal = self.anStftWrapper.inverseOneSidedStft(signal=oneSidedStft, windowLength=self.fft_window_length,
                                                                hopSize=self.fft_hop_size)
            self.assertEquals(len(timeSignal), self.signal_length)
        test03(self.fake_signal)
        test03(self.fake_noise)

    def test04TheOneSidedStftReturnsComplexCoefficients(self):
        def test04(signal):
            oneSidedStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                           hopSize=self.fft_hop_size)
            self.assertTrue(np.iscomplexobj(oneSidedStft))
        test04(self.fake_signal)
        test04(self.fake_noise)

    def test05TheMagAndPhaseOneSidedStftReturnsMagAndPhase(self):
        def test05(signal):
            magStft, phaseStft = self.anStftWrapper.magAndPhaseOneSidedStft(signal=signal,
                                                                            windowLength=self.fft_window_length,
                                                                            hopSize=self.fft_hop_size)
            complexStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                          hopSize=self.fft_hop_size)
            reComplexStft = magStft * np.exp(1.0j * phaseStft)
            np.testing.assert_allclose(complexStft, reComplexStft, rtol=1e-12)
        test05(self.fake_signal)
        test05(self.fake_noise)

    def test06TheLog10MagAndPhaseOneSidedDGTReturnsLog10ManAndPhase(self):
        def test06(signal):
            log10MagStft, phaseStft = self.anStftWrapper.log10MagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size)
            complexStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                          hopSize=self.fft_hop_size)
            reComplexStft = (10 ** log10MagStft) * np.exp(1.0j * phaseStft)
            np.testing.assert_allclose(complexStft, reComplexStft, rtol=1e-12, atol=1e-12)
        test06(self.fake_signal)
        test06(self.fake_noise)

    def test07ApplyingAClipToTheLog10MagAndPhaseOneSidedSTFT(self):
        def test07(signal):
            clipBelow = 1e-5
            clippedLog10MagStft, phaseStft = self.anStftWrapper.log10MagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size,
                                                                                      clipBelow=clipBelow)
            complexStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                          hopSize=self.fft_hop_size)
            reComplexStft = (10 ** clippedLog10MagStft) * np.exp(1.0j * phaseStft)
            np.testing.assert_allclose(complexStft, reComplexStft, rtol=1e-5, atol=1e-5)
        test07(self.fake_signal)
        test07(self.fake_noise)

    def test08ALoggedSpectrogramCanBeReconstructed(self):
        def test08(signal):
            log10MagStft, phaseStft = self.anStftWrapper.log10MagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size)
            reconstructedSignal = self.anStftWrapper.reconstructSignalFromLogged10Spectogram(logSpectrogram=log10MagStft,
                                                                                         phase=phaseStft,
                                                                                         windowLength=self.fft_window_length,
                                                                                         hopSize=self.fft_hop_size)

            np.testing.assert_allclose(reconstructedSignal, signal, rtol=1e-12, atol=1e-12)
        test08(self.fake_signal)
        test08(self.fake_noise)

    def test09AClippedLoggedSpectrogramCanBeReconstructed(self):
        def test09(signal):
            clipBelow = 1e-5
            log10MagStft, phaseStft = self.anStftWrapper.log10MagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size,
                                                                                      clipBelow=clipBelow)
            reconstructedSignal = self.anStftWrapper.reconstructSignalFromLogged10Spectogram(logSpectrogram=log10MagStft,
                                                                                         phase=phaseStft,
                                                                                         windowLength=self.fft_window_length,
                                                                                         hopSize=self.fft_hop_size)

            np.testing.assert_allclose(reconstructedSignal, signal, rtol=1e-4, atol=1e-4)
            self.assertTrue(np.linalg.norm(reconstructedSignal - signal) < 7e-3)
        test09(self.fake_signal)
        test09(self.fake_noise)

    def test10TheLogMagAndPhaseOneSidedDGTReturnsLogManAndPhase(self):
        def test10(signal):
            logMagStft, phaseStft = self.anStftWrapper.logMagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size)
            complexStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                          hopSize=self.fft_hop_size)
            reComplexStft = (np.e ** logMagStft) * np.exp(1.0j * phaseStft)
            np.testing.assert_allclose(complexStft, reComplexStft, rtol=1e-12, atol=1e-12)
        test10(self.fake_signal)
        test10(self.fake_noise)

    def test11ApplyingAClipToTheLogMagAndPhaseOneSidedSTFT(self):
        def test11(signal):
            clipBelow = np.e**-10
            clippedLogMagStft, phaseStft = self.anStftWrapper.logMagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size,
                                                                                      clipBelow=clipBelow)
            complexStft = self.anStftWrapper.oneSidedStft(signal=signal, windowLength=self.fft_window_length,
                                                          hopSize=self.fft_hop_size)
            reComplexStft = (np.e ** clippedLogMagStft) * np.exp(1.0j * phaseStft)
            np.testing.assert_allclose(complexStft, reComplexStft, rtol=1e-4, atol=1e-4)
        test11(self.fake_signal)
        test11(self.fake_noise)

    def test12ALoggedSpectrogramCanBeReconstructed(self):
        def test12(signal):
            logMagStft, phaseStft = self.anStftWrapper.logMagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size)
            reconstructedSignal = self.anStftWrapper.reconstructSignalFromLoggedSpectogram(logSpectrogram=logMagStft,
                                                                                         phase=phaseStft,
                                                                                         windowLength=self.fft_window_length,
                                                                                         hopSize=self.fft_hop_size)

            np.testing.assert_allclose(reconstructedSignal, signal, rtol=1e-12, atol=1e-12)
        test12(self.fake_signal)
        test12(self.fake_noise)

    def test13AClippedLoggedSpectrogramCanBeReconstructed(self):
        def test13(signal):
            clipBelow = np.e**-10
            logMagStft, phaseStft = self.anStftWrapper.logMagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size,
                                                                                      clipBelow=clipBelow)
            self.assertTrue(np.min(logMagStft) >= np.log(clipBelow))
            reconstructedSignal = self.anStftWrapper.reconstructSignalFromLoggedSpectogram(logSpectrogram=logMagStft,
                                                                                         phase=phaseStft,
                                                                                         windowLength=self.fft_window_length,
                                                                                         hopSize=self.fft_hop_size)

            np.testing.assert_allclose(reconstructedSignal, signal, rtol=9e-3, atol=9e-3)
            self.assertTrue(np.linalg.norm(reconstructedSignal - signal) < 7e-3)
        test13(self.fake_signal)
        test13(self.fake_noise)

    def test14ALLoggedSpectrogramCanBeNormalizeToHaveMaxValueEqualToZero(self):
        def test14(signal):
            nonNormalizedLogMagStft, phaseStft = self.anStftWrapper.logMagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size)
            self.assertFalse(np.max(nonNormalizedLogMagStft)==0)

            normalizedLogMagStft, phaseStft = self.anStftWrapper.logMagAndPhaseOneSidedStft(signal=signal,
                                                                                      windowLength=self.fft_window_length,
                                                                                      hopSize=self.fft_hop_size,
                                                                                      normalize=True)
            self.assertTrue(np.max(normalizedLogMagStft)==0)
        test14(self.fake_signal)
        test14(self.fake_noise)
