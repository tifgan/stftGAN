import tensorflow as tf
from tfnntools.utils import arg_helper
import yaml
from copy import deepcopy

def rprint(msg, reuse=False):
    """Print message only if reuse is False.
    If a block is being resued, its description will not be re-printed.
    """
    if not reuse:
        print(msg)

class BaseNet(object):
    """Base Net abstract class."""
    def default_params(self):
        d_params = dict()
        return d_params

    def __init__(self, params={}, name="BaseNet", debug_mode=False):
        self._debug_mode=debug_mode
        if self._debug_mode:
            print('User parameters...')
            print(yaml.dump(params))
        self._params = deepcopy(arg_helper(params, self.default_params()))
        if self._debug_mode:
            print('\nParameters used for the network...')
            print(yaml.dump(self._params))
        self._name = name
        self._outputs = None
        self._inputs = None
        self._loss = None
        self._build_net()
        self._add_summary()

    def _build_net(self, inputs):
        raise ValueError('This is an abstract class')

    def batch2dict(self, inputs):
        raise ValueError('This is an abstract class')

    def _add_summary(self):
        tf.summary.scalar('train/loss', self._loss,  collections=["train"])

    @property
    def name(self):
        return self._name

    @property
    def loss(self):
        return self._loss

    @property
    def outputs(self):
        return self._outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def params(self):
        return self._params

    @staticmethod
    def batch2dict(batch):
        raise NotImplementedError("This is a an abstract class.")
