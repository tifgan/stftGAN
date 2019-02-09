if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

import numpy as np

from gantools import data
from gantools.data.Dataset import slice_2d_patch, slice_2d


class TestSlice2d(unittest.TestCase):
    def test_general_patch(self):
        for spix in [2, 4, 8, 32, 64, 128, 256]:

            images = np.random.randn(100, 256, 256).astype(np.float32)

            simg = slice_2d_patch(images[0], spix=spix)

            # Testing
            img = images[0]
            sx, sy = img.shape
            nx = sx // spix
            ny = sy // spix
            img_test = np.zeros([nx * ny, spix, spix, 4], dtype=np.float32)
            for i in range(nx):
                for j in range(ny):
                    index = nx * j + i
                    img_test[index, :, :, 0] = img[i * spix:(
                        i + 1) * spix, j * spix:(j + 1) * spix]
                    if i:
                        img_test[index, :, :, 1] = img[(
                            i - 1) * spix:i * spix, j * spix:(j + 1) * spix]
                    if j:
                        img_test[index, :, :, 2] = img[i * spix:(
                            i + 1) * spix, (j - 1) * spix:j * spix]
                    if i and j:
                        img_test[index, :, :, 3] = img[(
                            i - 1) * spix:i * spix, (j - 1) * spix:j * spix]

            assert (np.sum(
                np.abs(img_test[:, :, :, 0] - simg[:, :, :, 0])) == 0)
            assert (np.sum(
                np.abs(img_test[:, :, :, 1] - simg[:, :, :, 1])) == 0)
            assert (np.sum(
                np.abs(img_test[:, :, :, 2] - simg[:, :, :, 2])) == 0)
            assert (np.sum(
                np.abs(img_test[:, :, :, 3] - simg[:, :, :, 3])) == 0)

            if spix == 64:
                index = 5
                img_d = np.concatenate(
                    [img_test[index, :, :, 1], img_test[index, :, :, 0]],
                    axis=0)
                img_u = np.concatenate(
                    [img_test[index, :, :, 3], img_test[index, :, :, 2]],
                    axis=0)
                img_r = np.concatenate([img_u, img_d], axis=1)

                np.testing.assert_array_equal(img_r,
                                              img[:2 * spix, :2 * spix])


    def test_slice_2d_simple(self):
        nx = 4
        nc = 2
        test_img = np.random.randn(nc, nx, nx).astype(np.float32)
        spix = 2
        sp = nx // spix
        res = []

        for i in range(nc):
            v = np.array(np.split(test_img[i, :, :], sp, axis=0))
            res.append(v)
        res = np.vstack(res)
        res = np.vstack(np.split(res, sp, axis=2))

        res2 = slice_2d(test_img, spix=spix)

        order = [0, 2, 1, 3, 4, 6, 5, 7]
        np.testing.assert_array_equal(res[order], res2)

    


if __name__ == '__main__':
    unittest.main()