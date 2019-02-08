import ltfatpy
from unittest import TestCase

import numpy as np

from modGabPhaseGrad import modgabphasegrad
ltfatpy.gabphasegrad = modgabphasegrad

__author__ = 'Andres'


class TestModGabPhaseGrad(TestCase):
    @classmethod
    def setUpClass(self):
        ltfatpy.gabphasegrad = modgabphasegrad
        self.signal_length = 5120
        self.sampling_rate = 44100
        self.fft_window_length = 512
        self.fft_hop_size = 128
        self.fake_time = np.arange(0, self.signal_length / self.sampling_rate, 1 / self.sampling_rate, dtype=np.float64)
        self.fake_signal = np.sin(2 * np.pi * 440 * self.fake_time, dtype=np.float64)
        self.fake_noise = np.random.normal(0, 1, size=self.signal_length)

    def test01dgtAndRealDgtObtainApproxTheSameResultsForThePhaseGrad(self):
        def test01(signal, fft_window_length, fft_hop_size, window, rtolgrad=1e-8, atolgrad=1e-9):
            dgt = ltfatpy.dgt(signal, window, fft_hop_size, fft_window_length)[0]
            dgtmag = np.abs(dgt)
            dgtmask = np.ones_like(dgtmag)
            dgtmask[dgtmag < 1e-4] = 0

            dgtreal = ltfatpy.dgtreal(signal, window, fft_hop_size, fft_window_length)[0]
            dgtrealmag = np.abs(dgtreal)
            dgtrealmask = np.ones_like(dgtrealmag)
            dgtrealmask[dgtrealmag < 1e-4] = 0

            np.testing.assert_allclose(dgt[:self.fft_window_length // 2 + 1], dgtreal, rtol=1e-14, atol=1e-14)

            tgrad, fgrad = ltfatpy.gabphasegrad('phase', np.angle(dgt), fft_hop_size, fft_window_length) * dgtmask
            tgradreal, fgradreal = ltfatpy.gabphasegrad('phase', np.angle(dgtreal), fft_hop_size,
                                                        fft_window_length) * dgtrealmask
            np.testing.assert_allclose(tgrad[:self.fft_window_length // 2 + 1], tgradreal, rtol=rtolgrad, atol=atolgrad)
            np.testing.assert_allclose(fgrad[:self.fft_window_length // 2 + 1], fgradreal, rtol=rtolgrad, atol=atolgrad)

        test01(self.fake_signal, self.fft_window_length, self.fft_hop_size, {'name': 'gauss', 'M': 1})
        test01(self.fake_noise, self.fft_window_length, self.fft_hop_size,
               {'name': 'gauss', 'M': self.fft_window_length})
        test01(self.fake_noise, self.fft_window_length, self.fft_hop_size,
               {'name': 'hann', 'M': self.fft_window_length})
        test01(self.fake_signal, self.fft_window_length, self.fft_hop_size, {'name': 'hann', 'M': 3})
