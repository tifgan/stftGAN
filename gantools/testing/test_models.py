if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.model import CosmoWGAN, WGAN, LapWGAN, UpscalePatchWGAN, UpscalePatchWGANBorders
from gantools.gansystem import GANsystem
from gantools.data.Dataset import Dataset
import numpy as np


class TestGANmodels(unittest.TestCase):
    def test_default_params_wgan(self):
        obj = GANsystem(WGAN)
        obj = GANsystem(CosmoWGAN)

    def test_default_params_lapgan(self):
        obj = GANsystem(LapWGAN)
        class UgradedGAN(LapWGAN, CosmoWGAN):
            pass
        obj = GANsystem(UgradedGAN)

    def test_default_params_patchgan(self):

        obj = GANsystem(UpscalePatchWGAN)
        class UgradedGAN(UpscalePatchWGAN, CosmoWGAN):
            pass
        obj = GANsystem(UgradedGAN)

    def test_default_params_patchgan_borders(self):
        obj = GANsystem(UpscalePatchWGANBorders)
        class UgradedGAN(UpscalePatchWGANBorders, CosmoWGAN):
            pass
        obj = GANsystem(UpscalePatchWGANBorders)

    def test_1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 1]  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 8]
        params['net']['generator']['nfilter'] = [2, 32, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn, bn]
        params['net']['generator']['shape'] = [[5], [5], [5], [5]]
        params['net']['generator']['stride'] = [1, 2, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5], [5], [5], [3]]
        params['net']['discriminator']['stride'] = [2, 2, 2, 1]
        params['net']['discriminator']['data_size'] = 1

        X = np.random.rand(101, 16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)

    def test_2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 16, 1]  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 8 * 8]
        params['net']['generator']['nfilter'] = [2, 32, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn, bn]
        params['net']['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5],
                                                   [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2, 1]
        params['net']['discriminator']['data_size'] = 2

        X = np.random.rand(101, 16, 16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 16, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)

    def test_3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 1]  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 4 * 4 * 4]
        params['net']['generator']['nfilter'] = [2, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [5, 5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5, 5], [3, 3, 3],
                                                   [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2]
        params['net']['discriminator']['data_size'] = 3

        X = np.random.rand(101, 8, 8, 8)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)


    def test_lapgan1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 1]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 16 * 16
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[5], [5], [5]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upsampling'] = 2

        X = np.random.rand(101, 16)
        dataset = Dataset(X)
        wgan = GANsystem(LapWGAN, params)
        wgan.train(dataset)
        X_down = np.random.rand(500, 8, 1)
        img = wgan.generate(N=2, X_down=X_down[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 1))
        img = wgan.generate(N=500, X_down=X_down[:500])
        assert (len(img) == 500)

    def test_lapgan2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 16, 1]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 16 * 16
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[5, 5], [5, 5], [5, 5]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 2
        params['net']['upsampling'] = 2

        X = np.random.rand(101, 16, 16)
        dataset = Dataset(X)

        class UgradedGAN(LapWGAN, CosmoWGAN):
            pass

        wgan = GANsystem(UgradedGAN, params)
        wgan.train(dataset)
        X_down = np.random.rand(500, 8, 8, 1)
        img = wgan.generate(N=2, X_down=X_down[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 16, 1))
        img = wgan.generate(N=500, X_down=X_down[:500])
        assert (len(img) == 500)

    def test_lapgan3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 1]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [3, 3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 3
        params['net']['upsampling'] = 2

        X = np.random.rand(101, 8, 8, 8)
        dataset = Dataset(X)

        class UgradedGAN(LapWGAN, CosmoWGAN):
            pass

        wgan = GANsystem(UgradedGAN, params)
        wgan.train(dataset)
        X_down = np.random.rand(500, 4, 4, 4, 1)
        img = wgan.generate(N=2, X_down=X_down[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500])
        assert (len(img) == 500)

    def test_patchgan1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 2]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3], [3], [3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upsampling'] = None

        X = np.random.rand(101, 8, 2)
        dataset = Dataset(X)
        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 1)
        img = wgan.generate(N=2, borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 1))
        img = wgan.generate(N=500, borders=borders[:500])
        assert (len(img) == 500)

    def test_patchgan2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 4]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3], [3, 3], [3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 2
        params['net']['upsampling'] = None

        X = np.random.rand(101, 8, 8, 4)
        dataset = Dataset(X)

        class UgradedGAN(UpscalePatchWGAN, CosmoWGAN):
            pass

        wgan = GANsystem(UgradedGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 3)
        img = wgan.generate(N=2, borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 1))
        img = wgan.generate(N=500, borders=borders[:500])
        assert (len(img) == 500)

    def test_patchgan3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 8]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [3, 3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 3
        params['net']['upsampling'] = None

        X = np.random.rand(101, 8, 8, 8, 8)
        dataset = Dataset(X)

        class UgradedGAN(UpscalePatchWGAN, CosmoWGAN):
            pass

        wgan = GANsystem(UgradedGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 8, 7)
        img = wgan.generate(N=2, borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(N=500, borders=borders[:500])
        assert (len(img) == 500)

    def test_patchupscalegan1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 2]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3], [3], [3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upsampling'] = 2

        X = np.random.rand(101, 8, 2)
        dataset = Dataset(X)
        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 1)
        X_down = np.random.rand(500, 4, 1)
        img = wgan.generate(N=2, X_down=X_down[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500], borders=borders[:500])
        assert (len(img) == 500)

    def test_patchupscalegan2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 4]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3], [3, 3], [3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 2
        params['net']['upsampling'] = 2

        X = np.random.rand(101, 8, 8, 4)
        dataset = Dataset(X)

        class UgradedGAN(UpscalePatchWGAN, CosmoWGAN):
            pass

        wgan = GANsystem(UgradedGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 3)
        X_down = np.random.rand(500, 4, 4, 1)
        img = wgan.generate(N=2, X_down=X_down[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500], borders=borders[:500])
        assert (len(img) == 500)

    def test_patchupscalegan3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 8]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [3, 3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 3
        params['net']['upsampling'] = 2

        X = np.random.rand(101, 8, 8, 8, 8)
        dataset = Dataset(X)

        class UgradedGAN(UpscalePatchWGAN, CosmoWGAN):
            pass

        wgan = GANsystem(UgradedGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 8, 7)
        X_down = np.random.rand(500, 4, 4, 4, 1)

        img = wgan.generate(N=2, X_down=X_down[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500], borders=borders[:500])
        assert (len(img) == 500)


    def test_patchupscalegan1dborder(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 2]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8
        params['net']['generator']['full'] = [16]
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3], [3], [3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['generator']['borders'] = dict()
        params['net']['generator']['borders']['width_full'] = None
        params['net']['generator']['borders']['nfilter'] = [4, 1]
        params['net']['generator']['borders']['batch_norm'] = [bn, bn]
        params['net']['generator']['borders']['shape'] = [[5], [3]]
        params['net']['generator']['borders']['stride'] = [2, 2]
        params['net']['generator']['borders']['data_size'] = 1
        params['net']['generator']['borders']['width_full'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upsampling'] = 2


        X = np.random.rand(101, 8, 2)
        dataset = Dataset(X)
        wgan = GANsystem(UpscalePatchWGANBorders, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 4, 1)
        X_smooth = np.random.rand(500, 4, 1)
        img = wgan.generate(N=2, X_smooth=X_smooth[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (4, 1))
        img = wgan.generate(N=500, X_smooth=X_smooth[:500], borders=borders[:500])
        assert (len(img) == 500)

if __name__ == '__main__':
    unittest.main()