if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest
import numpy as np
import tensorflow as tf
from gantools.plot.plot_summary import PlotSummary, PlotSummaryLog


class TestPlot(unittest.TestCase):
    def test_plot_summary(self):
        print('Testing the plot module')
        obj = PlotSummary('Objname', 'ObjCat')
        with tf.Session() as sess:
            test = obj.produceSummaryToWrite(sess)
        print('Test 1 done!')
        N = 10
        x = np.linspace(1, N, N)
        y1 = np.random.rand(N)
        y2 = np.random.rand(N)
        obj = PlotSummaryLog('Objname', 'ObjCat')
        with tf.Session() as sess:
            test = obj.produceSummaryToWrite(sess, x, y1, y2)
        print('Test 2 done!')

if __name__ == '__main__':
    unittest.main()