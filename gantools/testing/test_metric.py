if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import unittest

import numpy as np
from gantools.metric import ganlist
from gantools import metric
from scipy import stats


def wasserstein_distance_jonathan(x_og, y, w):
    assert (x_og.shape == y.shape == w.shape)
    x = np.copy(x_og)
    loss = 0
    for idx in range(x.shape[0] - 1):
        d = y[idx] - x[idx]
        x[idx] = x[idx] + d
        x[idx + 1] = x[idx + 1] - d
        loss = loss + np.abs(d * (w[idx + 1] - w[idx]))
    return loss / (w[-1] - w[0])


class TestMetric(unittest.TestCase):
    def test_wasserstein_distance(self):
        w = np.arange(0, 10)
        x = np.arange(0, 10, 1)
        y = np.arange(9, -1, -1)

        x = x / np.sum(np.abs(x))
        y = y / np.sum(np.abs(y))

        a = metric.wasserstein_distance(x, y, w)
        b = wasserstein_distance_jonathan(x, y, w)

        np.testing.assert_almost_equal(a, b)

        w = np.cumsum(np.random.rand(10))
        x = np.random.rand(10)
        y = np.random.rand(10)

        x = x / np.sum(np.abs(x))
        y = y / np.sum(np.abs(y))

        a = metric.wasserstein_distance(x, y, w)
        b = wasserstein_distance_jonathan(x, y, w)

        np.testing.assert_almost_equal(a, b)

    def test_gan_stat_list(self):
        l = ganlist.gan_stat_list()
        x = np.random.randn(100, 20, 50) + 7
        for s in l:
            s(x)
        _, limx, meanx, varx, skewx, kurtx = stats.describe(x.flatten())
        minx, maxx = limx
        np.testing.assert_almost_equal(meanx, l[0](x))
        np.testing.assert_almost_equal(varx, l[1](x), decimal=3)
        np.testing.assert_almost_equal(minx, l[2](x))
        np.testing.assert_almost_equal(maxx, l[3](x))
        np.testing.assert_almost_equal(kurtx, l[4](x))
        np.testing.assert_almost_equal(skewx, l[5](x))

    def test_gan_metric_list(self):
        metric_list = ganlist.gan_metric_list(recompute_real=True)
        x = np.random.randn(100, 20, 50) + 7
        y = 2 * np.random.randn(100, 20, 50) + 8
        for m in metric_list[:7]:
            print(m.group+'/'+m.name)
            m(x, y)
            assert (m(y, y) == 0)
            assert (m(x, x) == 0)
        x = np.abs(np.random.randn(100, 32, 32) + 7)
        y = np.abs(2 * np.random.randn(100, 32, 32) + 8)
        for m in metric_list:
            print(m.group+'/'+m.name)
            m(x, y)
            assert (m(y, y) == 0)
            assert (m(x, x) == 0)

        x = np.ones(20)
        y = 2 * np.ones(20)
        assert (metric_list[0](x, y) == 1)


if __name__ == '__main__':
    unittest.main()