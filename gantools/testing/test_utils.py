if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

import numpy as np

from gantools import utils
import tensorflow as tf

class TestUtils(unittest.TestCase):
    def test_tf_cube_slices(x):
        img = tf.placeholder(tf.float32,shape=[None, 32,32,32,1])
        utils.tf_cube_slices(img)
        img = tf.placeholder(tf.float32,shape=[None, 4,4,4,1])
        utils.tf_cube_slices(img)
    def test_get_closest_divisor(x):

        assert(utils.get_closest_divisor(4)==2)
        assert(utils.get_closest_divisor(16)==4)
        assert(utils.get_closest_divisor(9)==3)
        assert(utils.get_closest_divisor(2)==1)
        assert(utils.get_closest_divisor(3)==3)
        assert(utils.get_closest_divisor(35)==7)
        for i in range(100):
            assert(np.mod(i,utils.get_closest_divisor(i))==0)

if __name__ == '__main__':
    unittest.main()