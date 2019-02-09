if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import unittest

import numpy as np

from gantools.data.transformation import *

import tensorflow as tf
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestSlice(unittest.TestCase):
    def test_patch2img_1d(self):
        imgs = np.random.rand(25, 64)
        patches = slice_1d_patch(imgs, spix=32)
        imgs2 = patch2img(patches, size=1)
        np.testing.assert_almost_equal(imgs2[-25:], imgs)

    def test_patch2img_2d(self):
        imgs = np.random.rand(25, 64, 64)
        patches = slice_2d_patch(imgs, spix=32)
        imgs2 = patch2img(patches, size=2)
        np.testing.assert_almost_equal(imgs2[-25:], imgs)

    def test_patch2img_3d(self):
        imgs = np.random.rand(25, 64, 64, 64)
        patches = slice_3d_patch(imgs, spix=32)
        imgs2 = patch2img(patches, size=3)
        np.testing.assert_almost_equal(imgs2[-25:], imgs)

    def test_tf_patch2img_1d(self):
        imgs = np.random.rand(25, 64)
        patches = slice_1d_patch(imgs, spix=32)
        imgs2 = patch2img(patches, size=1)
        args = (patches[:, :, 0], patches[:, :, 1])
        with tf.Session() as sess:
            imgs3 = tf_patch2img_1d(*args).eval()
        np.testing.assert_almost_equal(imgs3, imgs2)

    def test_tf_patch2img_2d(self):
        imgs = np.random.rand(25, 64, 64)
        patches = slice_2d_patch(imgs, spix=32)
        imgs2 = patch2img(patches, size=2)
        args = (patches[:, :, :, 0], patches[:, :, :, 1], patches[:, :, :, 2],
                patches[:, :, :, 3])
        with tf.Session() as sess:
            imgs3 = tf_patch2img_2d(*args).eval()
        np.testing.assert_almost_equal(imgs3, imgs2)

    def test_tf_patch2img_3d(self):
        imgs = np.random.rand(25, 16, 16, 16)
        patches = slice_3d_patch(imgs, spix=8)
        imgs2 = patch2img(patches, size=3)
        args = (patches[:, :, :, :, 0], patches[:, :, :, :, 1],
                patches[:, :, :, :, 2], patches[:, :, :, :, 3],
                patches[:, :, :, :, 4], patches[:, :, :, :, 5],
                patches[:, :, :, :, 6], patches[:, :, :, :, 7])
        with tf.Session() as sess:
            imgs3 = tf_patch2img_3d(*args).eval()
        np.testing.assert_almost_equal(imgs3, imgs2)

    def test_flip_slices_1d(self):
        imgs = np.zeros([25, 16])
        imgs[:, 4:12] = 1
        patches = slice_1d_patch(imgs, spix=8)[-25:]
        r,l = patches[:, :, 0:1], patches[:, :, 1:2]
        flip_l = flip_slices_1d(l)
        np.testing.assert_almost_equal(r, flip_l)


    def test_flip_slices_2d(self):
        imgs = np.zeros([25, 16, 16])
        imgs[:, 4:12, 4:12] = 1
        patches = slice_2d_patch(imgs, spix=8)[-25:]
        dr, dl, ur, ul = patches[:, :, :, 0:
                                 1], patches[:, :, :, 1:
                                             2], patches[:, :, :, 2:
                                                         3], patches[:, :, :,
                                                                     3:4]
        flip_dl, flip_ur, flip_ul = flip_slices_2d(dl, ur, ul)
        np.testing.assert_almost_equal(dr, flip_dl)
        np.testing.assert_almost_equal(dr, flip_ur)
        np.testing.assert_almost_equal(dr, flip_ul)

    def test_flip_slices_3d(self):
        imgs = np.zeros([25, 16, 16, 16])
        imgs[:, 4:12, 4:12, 4:12] = 1
        patches = slice_3d_patch(imgs, spix=8)[-25:]
        args = (patches[:, :, :, :, 0:1], patches[:, :, :, :, 1:2],
                patches[:, :, :, :, 2:3], patches[:, :, :, :, 3:4],
                patches[:, :, :, :, 4:5], patches[:, :, :, :, 5:6],
                patches[:, :, :, :, 6:7], patches[:, :, :, :, 7:8])
        sols = flip_slices_3d(*args[1:])
        for sol in sols:
            np.testing.assert_almost_equal(args[0], sol)
    
    def test_tf_flip_slice_1d(self):
        imgs = np.random.rand(25, 64)
        patches = slice_1d_patch(imgs, spix=32)
        a = flip_slices_1d(patches[:, :, 1])
        with tf.Session() as sess:
            a2t = tf_flip_slices_1d(patches[:, :, 1])
            a2 = a2t.eval()
        np.testing.assert_almost_equal(a, a2)


    def test_tf_flip_slice_2d(self):
        imgs = np.random.rand(25, 64, 64)
        patches = slice_2d_patch(imgs, spix=32)
        args = (patches[:, :, :, 1], patches[:, :, :, 2], patches[:, :, :, 3])
        a, b, c = flip_slices_2d(*args)

        with tf.Session() as sess:
            a2t, b2t, c2t = tf_flip_slices_2d(*args)
            a2, b2, c2 = a2t.eval(), b2t.eval(), c2t.eval()
        np.testing.assert_almost_equal(a, a2)
        np.testing.assert_almost_equal(b, b2)
        np.testing.assert_almost_equal(c, c2)

    def test_tf_flip_slice_3d(self):
        imgs = np.random.rand(25, 16, 16, 16)
        patches = slice_3d_patch(imgs, spix=8)
        args = (patches[:, :, :, :, 1], patches[:, :, :, :, 2],
                patches[:, :, :, :, 3], patches[:, :, :, :, 4],
                patches[:, :, :, :, 5], patches[:, :, :, :, 6],
                patches[:, :, :, :, 7])
        s1s = flip_slices_3d(*args)

        with tf.Session() as sess:
            tfst = tf_flip_slices_3d(*args)
            s2s = []
            for st in tfst:
                s2s.append(st.eval())
        for s1, s2 in zip(s1s, s2s):
            np.testing.assert_almost_equal(s1, s2)


if __name__ == '__main__':
    unittest.main()