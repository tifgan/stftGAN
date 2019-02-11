import numpy as np

__author__ = 'Andres'


class StrechableNumpyArray(object):
    """When trying to add values to a numpy array, things can get slow if the array is too large.
    This class tries to solve that by updating the size of the array incrementally"""
    def __init__(self, dtype=np.float32):
        self._dtype = dtype
        self.data = np.zeros((1000000,), dtype=self._dtype)
        self.size = 0

    def append(self, x):
        if self.size + len(x) >= len(self.data):
            capacity = 4 * len(self.data)
            newdata = np.zeros((capacity,), dtype=self._dtype)
            newdata[:self.size] = self.data[:self.size]
            self.data = newdata

        self.data[self.size: self.size + len(x)] = x
        self.size += len(x)

    def finalize(self):
        output_data = self.data[:self.size]
        del self.data
        return output_data
