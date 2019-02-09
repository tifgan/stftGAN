import numpy as np
import os
from gantools import utils
from gantools.data import path
from gantools.data import fmap
from gantools.data import transformation
from gantools.data.Dataset import Dataset_2d, Dataset_3d, Dataset_2d_patch, Dataset_3d_patch, Dataset_time, Dataset, DatasetPostTransform
from skimage import io
from functools import partial

from gantools import blocks



def load_samples_raw(nsamples=None, resolution=256, Mpch=70):
    ''' Load 2D or 3D raw images

    Arguments
    ---------
    * nsamples : desired number of samples (if None: all of them)
    * resolution : [256, 512]
    * Mpch : [70, 350]

    '''
    rootpath = path.root_path()
    input_pattern = '{}_nbody_{}Mpc'.format(resolution, Mpch)
    file_ext = '.h5'
    queue = []
    for file in os.listdir(rootpath):
        if file.endswith(file_ext) and input_pattern in file:
            queue.append(os.path.join(rootpath, file))
            # if len(queue) == 10:
            #     break

    if len(queue) == 0:
        raise LookupError('No file founds, check path and parameters')
    raw_images = []
    for file_path in queue:
        raw_images.append(
            utils.load_hdf5(
                filename=file_path, dataset_name='data', mode='r'))
        if type(raw_images[-1]) is not np.ndarray:
            raise ValueError(
                "Data stored in file {} is not of type np.ndarray".format(
                    file_path))

    raw_images = np.array(raw_images).astype(np.float32)


    if nsamples is None:
        return raw_images
    else:
        if nsamples > len(raw_images):
            raise ValueError("Not enough sample")
        else:
            print('Select {} samples out of {}.'.format(
                nsamples, len(raw_images)))

        return raw_images[:nsamples]


def load_nbody_dataset(
        nsamples=None,
        resolution=256,
        Mpch=70,
        shuffle=True,
        forward_map = None,
        spix=128,
        augmentation=True,
        scaling=1,
        is_3d=False,
        patch=False):

    ''' Load a 2D or a 3D nbody images dataset:

     Arguments
    ---------
    * nsamples : desired number of samples, if None => all of them (default None)
    * resolution : resolution of the original cube [256, 512] (default 256)
    * Mpch : [70, 350] (default 70)
    * shuffle: shuffle the data (default True)
    * foward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * is_3d : load a 3d dataset (default False)
    * patch: experimental feature for patchgan
    '''

    # 1) Load raw images
    images = load_samples_raw(nsamples=nsamples, resolution=resolution, Mpch=Mpch)
    print("images shape = ", images.shape)

    # 2) Apply forward map if necessary
    if forward_map:
        images = forward_map(images)

    if (not is_3d):
        sh = images.shape
        images = images.reshape([sh[0]*sh[1], sh[2], sh[3]])

    # 2p) Apply downscaling if necessary
    if scaling>1:
        if is_3d:
            data_shape = 3
        else:
            data_shape = 2
        images = blocks.downsample(images, scaling, size=data_shape)

    if augmentation:
        # With the current implementation, 3d augmentation is not supported
        # for 2d scaling
        if is_3d:
            t = partial(transformation.random_transformation_3d, roll=True)
        else:
            t = partial(transformation.random_transformation_2d, roll=True)
    else:
        t = None
    
    # 5) Make a dataset
    if patch:
        if is_3d:
            dataset = Dataset_3d_patch(images, spix=spix, shuffle=shuffle, transform=t)
        else:
            dataset = Dataset_2d_patch(images, spix=spix, shuffle=shuffle, transform=t)

    else:
        if is_3d:
            dataset = Dataset_3d(images, spix=spix, shuffle=shuffle, transform=t)
        else:
            dataset = Dataset_2d(images, spix=spix, shuffle=shuffle, transform=t)

    return dataset


# def load_dataset_file(
#         nsamples=None,
#         resolution=256,
#         Mpch=70,
#         shuffle=True,
#         forward_map = None,
#         spix=128,
#         augmentation=True,
#         scaling=1,
#         is_3d=False,
#         patch=False):

#     ''' Load a 2D dataset object 

#      Arguments
#     ---------
#     * nsamples : desired number of samples, if None => all of them (default None)
#     * resolution : [256, 512] (default 256)
#     * Mpch : [70, 350] (default 70)
#     * shuffle: shuffle the data (default True)
#     * foward : foward mapping use None for raw data (default None)
#     * spix : resolution of the image (default 128)
#     * augmentation : use data augmentation (default True)
#     * scaling : downscale the image by a factor (default 1)
#     * is_3d : load a 3d dataset (default False)
#     * patch: experimental feature for patchgan
#     '''

#     if augmentation:
#         # With the current implementation, 3d augmentation is not supported
#         # for 2d scaling
#         if scaling>1 and not is_3d:
#             t = transformation.random_transformation_2d
#         else:
#             t = transformation.random_transformation_3d
#     else:
#         t = None
    
#     # 5) Make a dataset
#     if patch:
#         if is_3d:
#             dataset = Dataset_file_3d_patch(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling, 
#             spix=spix, shuffle=shuffle, transform=t)
#         else:
#             dataset = Dataset_file_2d_patch(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling, 
#             spix=spix, shuffle=shuffle, transform=t)

#     else:
#         if is_3d:
#             dataset = Dataset_file_3d(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling,
#             spix=spix, shuffle=shuffle, transform=t)
#         else:
#             dataset = Dataset_file_2d(resolution=resolution, Mpch=Mpch,
#             forward_map = forward_map, scaling=scaling,
#             spix=spix, shuffle=shuffle, transform=t)

#     return dataset



    
def load_time_dataset(
        resolution=256,
        Mpch=100,
        shuffle=True,
        forward_map = None,
        spix=128,
        augmentation=True):

    ''' Load a 2D dataset object 

     Arguments
    ---------
    * resolution : [256, 512] (default 256)
    * Mpch : [100, 500] (default 70)
    * shuffle: shuffle the data (default True)
    * foward : foward mapping use None for raw data (default None)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    '''

    # 1) Load raw images
    images = load_time_cubes(resolution=resolution, Mpch=Mpch)
    # (ts, resolution, resolution, resolution)

    # 2) Apply forward map if necessary
    if forward_map:
        images = forward_map(images)
    if augmentation:
        t = transformation.random_transformation_3d
    else:
        t = None

    # 5) Make a dataset
    dataset = Dataset_time(X=images, shuffle=shuffle, slice_fn=slice_fn, transform=transform)

    return dataset


def load_medical_data():
    pathdata = os.path.join(path.medical_path(),'volumedata.tif')
    return np.array(io.imread(pathdata))

def do_nothing(x):
    return x

def load_medical_dataset(
        shuffle=True,
        spix=32,
        augmentation=True,
        scaling=1,
        patch=False):

    ''' Load a 2D dataset object 

     Arguments
    ---------
    * shuffle: shuffle the data (default True)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * patch: experimental feature for patchgan
    '''

    images = load_medical_data()
    images = images.reshape([1, *images.shape])
    dtype = np.uint8
    if scaling>1:
        images = blocks.downsample(images, scaling, 3).astype(np.float32)
        dtype = np.float32
    post_transform = fmap.medical_forward

    if augmentation:
        if scaling==1:
            t = partial(transformation.random_transformation_3d, roll=False, spix=512)
        else:
            t = partial(transformation.random_transformation_3d, roll=False, spix=spix)
    else:
        t = do_nothing

    if patch:
        slice_fn = partial(transformation.slice_3d_patch, spix=spix)
    else:
        slice_fn = partial(transformation.slice_3d, spix=spix)
    dataset = DatasetPostTransform(images, slice_fn=slice_fn,
        shuffle=shuffle, transform=t, post_transform=post_transform,
        dtype=dtype)
    return dataset


def load_nysnth_rawdata():
    pathdata = os.path.join(path.nsynth_path(), 'nsynth-valid.npz')
    return np.load(pathdata)['arr_0']

def load_piano_rawdata():
    pathdata = os.path.join(path.piano_path(), 'piano-train.npz')
    return np.load(pathdata)['arr_0']


def load_audio_dataset(
        shuffle=True, scaling=1, patch=False, augmentation=False, spix=None, smooth=None, type='nsynth'):
    ''' Load a Nsynth dataset object.

     Arguments
    ---------
    * shuffle: shuffle the data (default True)
    * scaling : downscale the image by a factor (default 1)
    * path : downscale the image by a factor (default 1)
    * scaling : downscale the image by a factor (default 1)
    '''

    if type == 'nsynth':
        sig = load_nysnth_rawdata()
        sig = sig[:, :2**15]
    elif type == 'piano':
        sig = load_piano_rawdata()
    else:
        raise ValueError('Incorrect value for type')

    if len(sig.shape)==1:
        sig = np.reshape(sig, [1,len(sig)])

    # if augmentation and (not patch):
    #     raise ValueError('Augementation works only with patches.')
    
    # 1) Transform the data
    def transform(x):
        x = x/(2**15)
        x = (0.99*x.T/np.max(np.abs(x), axis=1)).T
        return x
    sig = transform(sig)


    # 2) Downsample
    Nwin = 32
    if scaling>1:
        # sig = blocks.downsample(sig, scaling)
        sig = transformation.downsample_1d(sig, scaling, Nwin=Nwin)

    if smooth is not None:
        sig = sig[:, :(sig.shape[1]//smooth)*smooth]
        sig_down = transformation.downsample_1d(sig, smooth, Nwin=Nwin)
        sig_smooth = transformation.upsamler_1d(sig_down, smooth, Nwin=Nwin)

        sig = np.concatenate((np.expand_dims(sig, axis=2), np.expand_dims(sig_smooth, axis=2)), axis=2)
    if patch:

        slice_fn = partial(transformation.slice_1d_patch, spix=spix)
    else:
        if spix is not None:
            slice_fn = partial(transformation.slice_1d, spix=spix)
        else:
            slice_fn = do_nothing

    if augmentation:
        transform = partial(transformation.random_shift_1d, roll=False, spix=spix)
    else:
        transform = do_nothing
    # 3) Make a dataset
    dataset = Dataset(sig, shuffle=shuffle, transform=transform, slice_fn=slice_fn)

    return dataset


def load_berlin_rawdata(nsamples=None):
    pathfolder = path.berlin_path()
    files = os.listdir(pathfolder)
    filepaths = []
    for file in files:
        if 'image.png' in file:
            filepaths.append(os.path.join(pathfolder, file))
    imgs = []
    if nsamples is None:
        nsamples = len(filepaths)
    for filepath in filepaths[:nsamples]:
        # Crop the data to have multiple of 256.
        imgs.append(io.imread(filepath)[:2304, :2560, :])
    imgs = np.array(imgs)
    return imgs


def load_maps_dataset(
        nsamples=None,
        shuffle=True,
        spix=128,
        augmentation=True,
        scaling=1,
        patch=False):

    ''' Load a 2D dataset object 

     Arguments
    ---------
    * nsamples : desired number of samples, if None => all of them (default None)
    * shuffle: shuffle the data (default True)
    * spix : resolution of the image (default 128)
    * augmentation : use data augmentation (default True)
    * scaling : downscale the image by a factor (default 1)
    * patch: experimental feature for patchgan

    The images comes from 
    https://zenodo.org/record/1154821#.W8WfkC0zbIo
    '''

    # 1) Load raw images
    images = load_berlin_rawdata(nsamples=nsamples)
    print("images shape = ", images.shape)


    # 2p) Apply downscaling if necessary
    dtype = np.uint8

    if scaling>1:
        data_shape = 2
        images = blocks.downsample(images, scaling, size=data_shape)
        dtype = np.float32
    if augmentation:
        if patch:
            t = partial(transformation.random_transformation_2d, roll=False, spix=2*spix)
        else:
            t = partial(transformation.random_transformation_2d, roll=False, spix=spix)
    else:
        t = None
    
    post_transform = fmap.medical_forward

    # 5) Make a dataset
    if patch:
        class SpecialDataset(DatasetPostTransform, Dataset_2d_patch):
            pass
        dataset = SpecialDataset(images, spix=spix, shuffle=shuffle, transform=t, post_transform=post_transform, dtype=dtype)
    else:
        class SpecialDataset(DatasetPostTransform, Dataset_2d):
            pass
        dataset = SpecialDataset(images, spix=spix, shuffle=shuffle, transform=t, post_transform=post_transform, dtype=dtype)
    return dataset
