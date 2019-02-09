if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

import numpy as np
from gantools.data import fmap
from gantools import data



class TestMetric(unittest.TestCase):

    def list_map_to_test(self):
        maps = []
        maps.append(('log_norm', fmap.log_norm_forward, fmap.log_norm_backward))
        maps.append(('gauss', fmap.gauss_forward, fmap.gauss_backward))
        maps.append(('log', fmap.log_forward, fmap.log_backward))
        maps.append(('nati', fmap.nati_forward, fmap.nati_backward))
        maps.append(('uniform', fmap.uniform_forward, fmap.uniform_backward))
        maps.append(('tanh', fmap.tanh_forward, fmap.tanh_backward))
        maps.append(('andres', fmap.andres_forward, fmap.andres_backward))
        maps.append(('stat', fmap.stat_forward, fmap.stat_backward))
        maps.append(('medical', fmap.medical_forward, fmap.medical_backward))
        maps.append(('default', fmap.forward, fmap.backward))
        return maps

    def test_mapos(self):
        dataset = data.load.load_nbody_dataset(nsamples=1, spix=32, resolution=256,Mpch=350)
        maps = self.list_map_to_test()
        X = dataset.get_all_data().flatten()
        for name, forward, backward in maps:
            print('Test map: {}'.format(name))
            x = forward(X)
            print(np.sum(np.abs(forward(backward(x))-x)))
            assert(np.sum(np.abs(forward(backward(x))-x))< 1)



if __name__ == '__main__':
    unittest.main()