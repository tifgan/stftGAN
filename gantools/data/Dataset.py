import itertools
import numpy as np
from gantools.utils import compose2
import functools
from .transformation import slice_2d, slice_3d, slice_3d_patch, slice_2d_patch, slice_time

def do_nothing(x):
    return x


class Dataset(object):
    ''' Dataset oject for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, shuffle=True, slice_fn=None, transform=None, dtype=np.float32):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.
        * slice_fn : Slicing function to cut the data into smaller parts
        '''
        X = X.astype(dtype)
        self._shuffle = shuffle
        if slice_fn:
            self._slice_fn = slice_fn
        else:
            self._slice_fn = do_nothing
        if transform:
            self._transform = transform
        else:
            self._transform = do_nothing

        self._data_process = compose2(self._transform, self._slice_fn)

        self._N = len(self._data_process(X))
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)
        self._X = X

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._data_process(self._X)[self._p]

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return self._data_process(self._X)[self._p[:N]]

    # TODO: kwargs to be removed
    def iter(self, batch_size=1, **kwargs):
        return self.__iter__(batch_size, **kwargs)

    # TODO: kwargs to be removed
    def __iter__(self, batch_size=1, **kwargs):

        if batch_size > self.N:
            raise ValueError(
                'Batch size greater than total number of samples available!')

        # Reshuffle the data
        if self.shuffle:
            self._p = np.random.permutation(self._N)
        nel = (self._N // batch_size) * batch_size
        transformed_data = self._data_process(self._X)[self._p[range(nel)]]
        for data in grouper(transformed_data, batch_size):
            yield np.array(data)

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N

class DatasetPostTransform(Dataset):
    def __init__(self, *args, post_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        if post_transform:
            self._post_transform = post_transform
        else:
            self._post_transform = do_nothing
    def __iter__(self, *args, **kwargs):
        it = super().__iter__(*args, **kwargs)
        for el in it:
            yield self._post_transform(el)

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._post_transform(super().get_all_data())

    def get_samples(self, *args, **kwargs):
        ''' Get the `N` first samples '''
        return self._post_transform(super().get_samples(*args, **kwargs))


class Dataset_3d(Dataset):
    def __init__(self, *args, spix=32, **kwargs):
        ''' Initialize a Dataset object for 3D images
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_3d, spix=spix)
        super().__init__(*args, slice_fn=slice_fn, **kwargs)



class Dataset_2d(Dataset):
    def __init__(self, *args, spix=128, **kwargs):
        ''' Initialize a Dataset object for 2D images
        '''

        slice_fn = functools.partial(slice_2d, spix=spix)
        super().__init__(*args, slice_fn=slice_fn, **kwargs)

class Dataset_time(Dataset):
    def __init__(self, X, spix=128, shuffle=True, transform=None):
        ''' Initialize a Dataset object
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_time, spix=spix)
        super().__init__(
            X=X, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

class Dataset_2d_patch(Dataset):
    def __init__(self, *args, spix=128, **kwargs):
        ''' Initialize a Dataset object for the 2d patch case
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_2d_patch, spix=spix)
        super().__init__(*args, slice_fn=slice_fn, **kwargs)


    def get_samples_full(self, N=100):
        X = self.get_samples(N=N)
        X_d = np.concatenate([X[:, :, :, 1], X[:, :, :, 0]], axis=1)
        X_u = np.concatenate([X[:, :, :, 3], X[:, :, :, 2]], axis=1)
        X_r = np.squeeze(np.concatenate([X_u, X_d], axis=2))
        return X_r


class Dataset_3d_patch(Dataset):
    def __init__(self, *args, spix=32, **kwargs):
        ''' Initialize a Dataset object for the 3d patch case
        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * transform : Function to be applied to each bigger cube in the dataset
                      for data augmentation
        '''

        slice_fn = functools.partial(slice_3d_patch, spix=spix)
        super().__init__(*args, slice_fn=slice_fn, **kwargs)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

