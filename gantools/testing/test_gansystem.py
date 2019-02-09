if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.model import WGAN
from gantools.gansystem import GANsystem, PaulinaGANsystem
from gantools.data.Dataset import Dataset
import numpy as np


class TestGANsystem(unittest.TestCase):
    def test_default_params(self):
        wgan = GANsystem(WGAN)
        wgan = PaulinaGANsystem(WGAN)


    def test_gansystem(self):
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        X = np.random.rand(101,16,16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        img = wgan.generate(500)
        assert(len(img)==500)

    def test_paulinasystem(self):
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        X = np.random.rand(101,16,16)
        dataset = Dataset(X)
        wgan = PaulinaGANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        img = wgan.generate(500)
        assert(len(img)==500)

    def test_generate(self):
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net']= dict()
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 5
        X = np.random.rand(101,16,16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        img = wgan.generate(500)
        assert(len(img)==500)
        z = np.random.randn(133,5)

        img1 = wgan.generate(z=z)

        img2 = np.zeros(shape=img1.shape)
        for i in range(133):
            img2[i] = wgan.generate(z=np.reshape(z[i], [1,5]))
        np.testing.assert_almost_equal(img1, img2, decimal=6)

if __name__ == '__main__':
    unittest.main()