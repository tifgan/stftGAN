if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.data import load, fmap
import numpy as np
from gantools.blocks import np_downsample_2d, np_downsample_3d, np_downsample_1d

class TestGANmodels(unittest.TestCase):
    def test_cosmo(self):
        forward = fmap.stat_forward
        dataset = load.load_nbody_dataset(
            nsamples=None, spix=32, Mpch=350, forward_map=forward, patch=True)
        it = dataset.iter(10)
        print(next(it).shape)
        assert (next(it).shape == (10, 32, 32, 4))
        del it, dataset

        dataset = load.load_nbody_dataset(
            nsamples=None,
            spix=32,
            Mpch=350,
            forward_map=forward,
            patch=True,
            is_3d=True)
        it = dataset.iter(4)
        print(next(it).shape)
        assert (next(it).shape == (4, 32, 32, 32, 8))
        del it, dataset

        dataset = load.load_nbody_dataset(
            nsamples=None, spix=32, Mpch=70, forward_map=None, patch=False)
        it = dataset.iter(10)
        print(next(it).shape)

        assert (next(it).shape == (10, 32, 32, 1))
        del it, dataset

        dataset = load.load_nbody_dataset(
            nsamples=2, spix=256, Mpch=70, forward_map=forward, patch=False)
        assert (dataset.get_all_data().shape[0] == 256 * 2)
        del dataset

        dataset = load.load_nbody_dataset(
            nsamples=2, spix=128, Mpch=350, forward_map=forward, patch=False)
        it = dataset.iter(10)
        print(next(it).shape)
        assert (next(it).shape == (10, 128, 128, 1))
        del it, dataset
        dataset1 = load.load_nbody_dataset(
            nsamples=4, spix=128, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2, is_3d=True)
        it1 = dataset1.iter(3)
        s1 = next(it1)
        del it1, dataset1

        dataset2 = load.load_nbody_dataset(
            nsamples=4, spix=32, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8, is_3d=True)
        it2 = dataset2.iter(3)
        s2 = next(it2)
        del it2, dataset2
        np.testing.assert_allclose(np_downsample_3d(s1,4), s2)

        dataset1 = load.load_nbody_dataset(
            nsamples=2, spix=128, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=2)
        it1 = dataset1.iter(10)
        s1 = next(it1)
        del it1, dataset1

        dataset2 = load.load_nbody_dataset(
            nsamples=2, spix=32, Mpch=350, forward_map=forward, patch=False, shuffle=False, augmentation=False, scaling=8)
        it2 = dataset2.iter(10)
        s2 = next(it2)
        del it2, dataset2
        np.testing.assert_allclose(np_downsample_2d(s1,4), s2)


    def test_maps(self):
        dataset = load.load_maps_dataset(spix=64, scaling=8, patch=False, augmentation=False)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 64, 64, 3))
        del it, dataset

        dataset = load.load_maps_dataset(spix=128, scaling=4, patch=False, augmentation=True)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 128, 128, 3))
        del it, dataset

        dataset = load.load_maps_dataset(spix=16, scaling=1, patch=True, augmentation=True)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 16, 16, 12))
        del it, dataset

        dataset = load.load_maps_dataset(
            spix=128, patch=False, shuffle=False, augmentation=False, scaling=2)
        it = dataset.iter(10)
        s1 = next(it)
        del it, dataset

        dataset = load.load_maps_dataset(
            spix=32, patch=False, shuffle=False, augmentation=False, scaling=8)
        it = dataset.iter(10)
        s2 = next(it)
        del it, dataset
        np.testing.assert_allclose(np_downsample_2d(s1,4), s2)

    def test_medical(self):
        dataset = load.load_medical_dataset(spix=32, scaling=8, patch=False, augmentation=False)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 32, 32, 32))
        del it, dataset

        dataset = load.load_medical_dataset(spix=32, scaling=8, patch=False, augmentation=True)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 32, 32, 32))
        del it, dataset

        dataset = load.load_medical_dataset(spix=16, scaling=8, patch=True, augmentation=True)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 16, 16, 16, 8))
        del it, dataset

        dataset = load.load_medical_dataset(
            spix=128, patch=False, shuffle=False, augmentation=False, scaling=2)
        it = dataset.iter(10)
        s1 = next(it)
        del it, dataset

        dataset = load.load_medical_dataset(
            spix=32, patch=False, shuffle=False, augmentation=False, scaling=8)
        it = dataset.iter(10)
        s2 = next(it)
        del it, dataset
        np.testing.assert_allclose(np_downsample_3d(s1,4), s2)


    def test_audio(self):
        dataset = load.load_audio_dataset(scaling=64*4)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 128))
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 512))
        Nel = dataset.N
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64*4, shuffle=False)
        it = dataset.iter(5)
        s1 = next(it)
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64, shuffle=False)
        it = dataset.iter(5)
        s2 = next(it)
        del it, dataset
        # This test do not work with the new upsampling/downsampling scheme
        # np.testing.assert_allclose(np_downsample_1d(s2,4), s1)

        dataset = load.load_audio_dataset(scaling=64, patch=True, augmentation=True, spix=128)
        it = dataset.iter(5)
        s = next(it)
        assert(s.shape == (5, 128, 2))
        assert(dataset.N==Nel*3)
        del it, dataset

        dataset = load.load_audio_dataset(scaling=32, patch=True, augmentation=True, spix=256)
        it = dataset.iter(5)
        s = next(it)
        assert(s.shape == (5, 256, 2))
        assert(dataset.N==Nel*3)
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64*4, type='piano', spix=128)
        it = dataset.iter(5)
        assert (next(it).shape == (5, 128))
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64, spix=512, type='piano')
        it = dataset.iter(5)
        assert (next(it).shape == (5, 512))
        Nel = dataset.N
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64*4, spix=128, shuffle=False, type='piano')
        it = dataset.iter(5)
        s1 = next(it)
        del it, dataset

        dataset = load.load_audio_dataset(scaling=64, spix=512, shuffle=False, type='piano')
        it = dataset.iter(5)
        s2 = next(it)
        del it, dataset
        # This test do not work with the new upsampling/downsampling scheme
        # np.testing.assert_allclose(np_downsample_1d(s2,4), s1)

        dataset = load.load_audio_dataset(scaling=64, spix=128, patch=True, augmentation=True, type='piano')
        it = dataset.iter(5)
        s = next(it)
        assert(s.shape == (5, 128, 2))
        del it, dataset

        dataset = load.load_audio_dataset(scaling=32, spix=256, patch=True, augmentation=True, type='piano')
        it = dataset.iter(5)
        s = next(it)
        assert(s.shape == (5, 256, 2))
        del it, dataset

if __name__ == '__main__':
    unittest.main()