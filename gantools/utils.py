"""Utility functions."""
import os
import functools
import shutil
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import h5py

from gantools.gansystem import GANsystem
import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile

def test_resume(try_resume, params):
    """ Try to load the parameters saved in `params['save_dir']+'params.pkl',`

        Not sure we should implement this function that way.
    """
    resume = False
    if try_resume:
        params_loaded = try_load_params(params['save_dir'])
        if params_loaded is None:
            print('No resume, the training will start from the beginning!')
        else:
            params = params_loaded
            print('Resume, the training will start from the last iteration!')
            resume = True
    return resume, params

def try_load_params(path):
    try:
        return load_params(path)
    except:
        return None

def load_params(path):
    with open(os.path.join(path,'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    return params

def load_gan(savepath, model, system=GANsystem):
    import gantools
    pathparams = os.path.join(savepath, 'params.pkl')
    with open(pathparams, 'rb') as f:          
        params = params = pickle.load(f)
    params['save_dir'] = savepath
    return system(model, params)



def sample_latent(m, n, prior="uniform", normalize=False):
    if prior == "uniform":
        return np.random.uniform(-1., 1., size=[m, n])
    elif prior == "gaussian":
        z = np.random.normal(0, 1, size=[m, n])
        if normalize:
            # Sample on the sphere
            z = (z.T * np.sqrt(n / np.sum(z * z, axis=1))).T
        return z
    elif prior.startswith('student'):
        prior_ = prior.split('-')
        if len(prior_) == 2:
            df = int(prior_[1])
        else:
            df = 3
        return np.random.standard_t(df, size=[m, n])
    elif prior == "laplacian":
        return np.random.laplace(loc=0.0, scale=1.0, size=[m, n])
    elif prior == "one-sided-laplacian":
        return np.abs(np.random.laplace(loc=0.0, scale=1.0, size=[m, n]))
    elif prior == "2-2tanh2":
        margin = 10*np.finfo(np.float32).eps
        u = np.random.uniform(low=0.0, high=1.0-margin, size=[m, n])
        return np.arctanh(0.5*(u+1.0))
    # elif prior.startswith('chi2'):
    #     prior_ = prior.split('-')
    #     if len(prior_) >= 2:
    #         df = int(prior_[1])
    #         if len(prior_) >= 3:
    #             k = float(prior_[2])
    #         else:
    #             k = 7
    #     else:
    #         df, k = 3, 7
    #     return simple_numpy(np.random.chisquare(df, size=[m, n]), k)
    # elif prior.startswith('gamma'):
    #     prior_ = prior.split('-')
    #     if len(prior_) >= 2:
    #         df = float(prior_[1])
    #         if len(prior_) >= 3:
    #             k = float(prior_[2])
    #         else:
    #             k = 4
    #     else:
    #         df, k = 1, 4
    #     return simple_numpy(np.random.gamma(df, size=[m, n]), k)
    else:
        raise ValueError(' [!] distribution not defined')


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def saferm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('Erase recursively directory: ' + path)
    if os.path.isfile(path):
        os.remove(path)
        print('Erase file: ' + path)


def makeit_square(x):
    sh = list(x.shape)
    nsamples, sx, sy = sh[0], sh[1], sh[2]
    if sx > sy:
        #         if mod(sx,sy):
        #             ValueError('Wrong size')
        cutx = sx // sy
        new_sh = sh
        new_sh[0] = nsamples * cutx
        new_sh[1] = sy
        new_x = np.zeros(new_sh)
        for i in range(cutx):
            new_x[i * nsamples:(i + 1) * nsamples, :, :] = x[:, i * sy:(
                i + 1) * sy, :]

    elif sy > sx:
        #         if mod(sy,sx):
        #             ValueError('Wrong size')
        cuty = sy // sx
        new_sh = sh
        new_sh[0] = nsamples * cuty
        new_sh[2] = sx
        new_x = np.zeros(new_sh)
        for i in range(cuty):
            new_x[i * nsamples:(i + 1) * nsamples, :, :] = x[:, :, i * sx:(
                i + 1) * sx]
    else:
        new_x = x
    return new_x


def get_tile_shape_from_3d_image(image_size):
    '''
    given a 3d image, tile it as a rectangle with slices of the 3d image,
    and return the shape of the rectangle
    '''
    l = len(image_size)

    if l == 3:
        x_dim, y_dim, z_dim = image_size
    elif l == 4:
        x_dim, y_dim, z_dim, _ = image_size
    else:
        raise ValueError("image_size too large!!")

    num_images_in_each_row = num_images_each_row(x_dim)
    tile_shape = ( y_dim * (x_dim//num_images_in_each_row), z_dim * num_images_in_each_row)
    return tile_shape


def num_images_each_row(x_dim):
    num_images_in_each_row = int(x_dim**0.5)
    while x_dim % num_images_in_each_row != 0:#smallest number that is larger than square root of x_dim and divides x_dim
        num_images_in_each_row += 1    

    return num_images_in_each_row


def tile_cube_slices(cubes):
    """
    cubes = [:, :, :, :]
    arrange each cube in cubes, as tile of squares
    """
    x_dim = cubes.shape[1]
    y_dim = cubes.shape[2]
    z_dim = cubes.shape[3]
    num_images_in_each_row = num_images_each_row(x_dim)

    tiles = []
    for cube in cubes:
        num = 0
        v_stacks = []
        for i in range(x_dim//num_images_in_each_row):
            h_stacks = []
            for j in range(num_images_in_each_row): # show 'num_images_in_each_row' squares from the cube in one row
                h_stacks.append(cube[num, :, :])
                num += 1
            v_stacks.append( np.hstack(h_stacks) )

        tile = np.vstack(v_stacks)
        tiles.append(tile.reshape([*(tile.shape), 1]))

    return np.array(tiles)

def get_closest_divisor(x):
    t = np.int(np.round(np.sqrt(x)))
    while np.mod(x, t):
        t += 1
    return t

def tf_cube_slices(cubes):
    """Takes stack of images as (?, w, h, num_images) and tiles them into a grid."""

    if len(cubes.shape) > 5:
        raise ValueError('To many dimensions')
    if len(cubes.shape) < 4:
        raise ValueError('To few dimensions')
    if len(cubes.shape) == 5:
        assert (cubes.shape[4] == 1)
        cubes = tf.squeeze(cubes, axis=[4])
    num_images = int(cubes.shape[3])
    num_rows = get_closest_divisor(num_images)
    # split last axis (num_images) into list of (h, w)
    cubes = tf.unstack(cubes, num=num_images, axis=-1)
    # tile all images horizontally into single row
    cubes = tf.concat(cubes, axis=2)
    # split into desired number of rows
    cubes = tf.split(cubes, num_rows, axis=2)
    # tile rows vertically
    cubes = tf.concat(cubes, axis=1)
    return cubes

def tf_make_grid(t, num_images, num_rows=2):
    '''takes stack of images as (?, w, h, num_images) and tiles them into a grid'''
    if len(t.shape)>5:
        raise ValueError('')
    t = tf.squeeze(t) # remove single batch, TODO make more flexible to work with higher batch size
    t = tf.unstack(t, num=num_images, axis=-1) # split last axis (num_images) into list of (h, w)
    t = tf.concat(t, axis=1) # tile all images horizontally into single row
    t = tf.split(t, num_rows, axis=1) # split into desired number of rows
    t = tf.concat(t, axis=0) # tile rows vertically
    return t


def get_3d_hists_dir_paths(path_3d_hists):
    dir_paths = []
    for item in os.listdir(path_3d_hists):
        dir_path = os.path.join(path_3d_hists, item)
        if os.path.isdir(dir_path) and item.endswith('hist'): # the directories where the 3d histograms are saved end with 'hist'
            dir_paths.append(dir_path)

    return dir_paths


# def make_batches(bs, *args):
#     """
#     Slide data with a size bs

#     Parameters
#     ----------
#     bs : batch size
#     *args : different pieces of data of the same size

#     """

#     ndata = len(args)
#     s0 = len(args[0])
#     for d in args:
#         if len(d) != s0:
#             raise ValueError("First dimensions differ!")

#     return itertools.zip_longest(
#         *(grouper(itertools.cycle(arg), bs) for arg in args))

# def grouper(iterable, n, fillvalue=None):
#     """
#     Collect data into fixed-length chunks or blocks. This function commes
#     from itertools
#     """
#     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
#     args = [iter(iterable)] * n
#     return itertools.zip_longest(fillvalue=fillvalue, *args)

def get_latent_dim(out_width, generator_params, is3d=False):
    """Calculate correct size for the latent dimension or fully connected layer
    before the first deconvolutional layer of a generator.

    Input arguments
    ---------------
    * out_width : output width of last deconvolutional layer of the network
    * generator_params  : parameter dict of the generator
    * is_3d  : whether 3d images should be generated by the network.

    Output: Required input layer size
    """
    w = out_width
    for stride in generator_params['stride']:
        w = w // stride
    if is3d:
        return w * w * w * generator_params['nfilter'][0]
    return w * w * generator_params['nfilter'][0]


def save_hdf5(data, filename, dataset_name='data', mode='w', dtype='float32'):
    h5f = h5py.File(filename, mode)
    h5f.create_dataset(dataset_name, data=data, dtype=dtype)
    h5f.close()


def load_hdf5(filename, dataset_name='data', mode='r'):
    h5f = h5py.File(filename, mode)
    data = h5f[dataset_name][:]
    h5f.close()
    return data


def load_dict_pickle(filename):
    with open(filename, 'rb') as infile:
        d = pickle.load(infile)
    return d


def save_dict_pickle(filename, dict_):
    with open(filename, 'wb') as outfile:
        pickle.dump(dict_, outfile)


def save_dict_for_humans(filename, dict_):
    """ Save dict in a pretty text format for humans. Cannot parse this back!
    Use save_dict_pickle for a load friendly version.
    """
    with open(filename, 'w') as outfile:

        outfile.write("All Params")
        outfile.write(str(dict_))

        if 'discriminator' in dict_:
            outfile.write("\nDiscriminator Params")
            outfile.write(str(dict_['discriminator']))

        if 'generator' in dict_:
            outfile.write("\nGenerator Params")
            outfile.write(str(dict_['generator']))

        if 'optimization' in dict_:
            outfile.write("\nOptimization Params")
            outfile.write(str(dict_['optimization']))

        if 'cosmology' in dict_:
            outfile.write("\nCosmology Params")
            outfile.write(str(dict_['cosmology']))

        if 'time' in dict_:
            outfile.write("\nTime Params")
            outfile.write(str(dict_['time']))


def compose2(first,second):
    """ Return the composed function `second(first(arg))` """
    return lambda x: second(first(x))


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def in_ipynb():
    """Test if the code is executed from ipython notebook.

    This function is not sure to be working properly.
    """
    try:
        cfg = get_ipython().config 
        return True
    except NameError:
        return False

def is_3d(img, is_square=True):
    """Infer from the data if the image is 3d. Return None if it does not know.

    Image format tested is [n, x, y, c] or [n, x, y, z, c], where
    * n is the number of images
    * x,y,z are the dimension of the image
    * c is the number of channels
    """
    if type(img) is tf.Tensor:
        sh = img.shape.as_list()
    else:
        sh = img.shape
    if len(sh)==5:
        return True
    if len(sh)==4:
        ls = sh[3]
        if ls==1:
            return False
        if is_square:
            if sh[2]==ls:
                return True
    return None


def get_data_size(img, is_square=True):
    """Infer from the data if the image is 3d, 2d or 1d. Return None if it does not know.

    Image format tested is [n, x, c], [n, x, y, c] or [n, x, y, z, c], where
    * n is the number of images
    * x,y,z are the dimension of the image
    * c is the number of channels
    """

    if type(img) is tf.Tensor:
        sh = img.shape.as_list()
    else:
        sh = img.shape
    if len(sh) == 5:
        return 3
    if len(sh) == 4:
        if is_square:
            if sh[3] == sh[2] == sh[1]:
                return 3
            elif sh[2] == sh[1]:
                return 2
            elif sh[2] == sh[3] == 1:
                return 1
        if sh[3] == 1:
            return 2
    if len(sh) == 3:
        if is_square:
            if sh[2] == sh[1]:
                return 2
            else:
                return 1
        if sh[2] == 1:
            return 1
    if len(sh) == 2:
        return 1
    return None


def get_lst_append_ret(params, key, obj):
    lst = params.get(key, list())
    lst.append(obj)
    return lst


def add_conv_layer(params, nfilter, shape=3, stride=1, batch_norm=False, is_3d=False):
    params['nfilter'] = get_lst_append_ret(params, 'nfilter', nfilter)
    params['stride'] = get_lst_append_ret(params, 'stride', stride)
    if batch_norm is not None:
        params['batch_norm'] = get_lst_append_ret(params, 'batch_norm', batch_norm)
    if not isinstance(shape, list):
        if is_3d:
            shape = [shape, shape, shape]
        else:
            shape = [shape, shape]
    params['shape'] = get_lst_append_ret(params, 'shape', shape)
    return params


def add_bottleneck_layer(params, nfilter, shape=3, stride=1, batch_norm=False, is_3d=False):
    assert isinstance(nfilter, list)
    assert len(nfilter) == 3
    params = add_conv_layer(params, nfilter[0], shape=1, stride=1,
                            batch_norm=batch_norm, is_3d=is_3d)
    params = add_conv_layer(params, nfilter[1], shape=shape, stride=stride,
                            batch_norm=batch_norm, is_3d=is_3d)
    params = add_conv_layer(params, nfilter[2], shape=1, stride=1,
                            batch_norm=batch_norm, is_3d=is_3d)
    return params


class NetParamHelper(object):
    def __init__(self, is_3d=False):
        self.params = dict()
        self.is_3d = False

    def add_conv_layer(self, nfilter, shape=3, stride=1, batch_norm=False):
        self.params = add_conv_layer(self.params, nfilter, shape=shape, stride=stride,
                                     batch_norm=batch_norm, is_3d=self.is_3d)

    def add_bottleneck_layer(self, nfilter, shape=3, stride=1, batch_norm=False):
        self.params = add_bottleneck_layer(self.params, nfilter, shape=shape, stride=stride,
                                     batch_norm=batch_norm, is_3d=self.is_3d)

    def add_full(self, units):
        self.params['full'] = get_lst_append_ret(self.params, 'full', units)

    def params(self):
        return self.params

def print_params_to_py_style_output_helper(name, params):
    print("\n# {} Params".format(name.title()))
    d_name = "params_{}".format(name)
    print("{} = dict()".format(d_name))
    for key, value in params.items():
        print("{}['{}'] = {}".format(d_name, key, value))
    print("params['{}'] = {}".format(name, d_name))


def print_params_to_py_style_output(params):
    print("# General Params")
    print("params = dict()")
    for key, value in params.items():
        if not isinstance(value, dict):
            print("params['{}'] = {}".format(key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_params_to_py_style_output_helper(key, value)


def print_sub_dict_params(d_name, params, indent = 0):
    indent_str = " " * indent
    print("\n{}{} params".format(indent_str, d_name).title())
    for key, value in params.items():
        if not isinstance(value, dict):
            print(" {}{}.{}: {}".format(indent_str, d_name, key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_sub_dict_params(key, value, indent=indent+1)


def print_param_dict(params):
    print("General Params")
    for key, value in params.items():
        if not isinstance(value, dict):
            print(" {}: {}".format(key, value))
    for key, value in params.items():
        if isinstance(value, dict):
            print_sub_dict_params(key, value)


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout