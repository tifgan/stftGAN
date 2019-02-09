if __name__ == '__main__':
	import sys, os
	sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

from gantools.testing import test_downsample
from gantools.testing import test_metric
from gantools.testing import test_split
from gantools.testing import test_slices
from gantools.testing import test_fmap
from gantools.testing import test_gansystem
from gantools.testing import test_models
from gantools.testing import test_utils
from gantools.testing import test_plots
from gantools.testing import test_load_data

loader = unittest.TestLoader()

suites = []
suites.append(loader.loadTestsFromModule(test_downsample))
suites.append(loader.loadTestsFromModule(test_metric))
suites.append(loader.loadTestsFromModule(test_split))
suites.append(loader.loadTestsFromModule(test_slices))
suites.append(loader.loadTestsFromModule(test_fmap))
suites.append(loader.loadTestsFromModule(test_gansystem))
suites.append(loader.loadTestsFromModule(test_utils))
suites.append(loader.loadTestsFromModule(test_models))
suites.append(loader.loadTestsFromModule(test_plots))
suites.append(loader.loadTestsFromModule(test_load_data))
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':  # pragma: no cover
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    run()