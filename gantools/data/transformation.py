import numpy as np
import tensorflow as tf
from scipy.signal import firwin


def random_shift_1d(signals, roll=False, spix=None, force_equal=True):
    """Apply a random shift to 1 d signal.

    If the roll option is not activated, the shift will not be circular and the
    data will be cropped accordingly.

    Arguments
    ---------
    roll: rolled shift (default False)
    spix: maximum size of the shift (if roll==False)
    force_equal: force all the return signals to be the same size using cropping (if roll==False)
    """
    nx = signals.shape[1]
    if roll:
        shiftx = np.random.randint(0, nx)
        out = np.roll(signals, shift=shiftx, axis=1)
    else:
        if spix is None:
            raise ValueError('spix needs to be specified.')
        shiftx = np.random.randint(0, spix)
        if force_equal:
            signals = signals[:,:(nx//spix)*spix]
            nx = signals.shape[1]

        if np.mod(nx, spix)==0:
            new_nx = ((nx//spix)-1)*spix
            out = signals[:, shiftx:shiftx+new_nx]
        else:
            out = signals[:, shiftx:]

    return out

def random_shift_2d(images, roll=False, spix=None):
    """Apply a random shift to 2d images.

    If the roll option is not activated, the shift will not be circular and the
    data will be cropped accordingly.

    Arguments
    ---------
    roll: rolled shift (default False)
    spix: maximum size of the shift (if roll==False)
    """
    nx = images.shape[1]
    ny = images.shape[2]
    if roll:
        shiftx = np.random.randint(0, nx)
        shifty = np.random.randint(0, ny)
        out = np.roll(images, shift=shiftx, axis=1)
        out = np.roll(out, shift=shifty, axis=2)
    else:
        if nx<spix or ny<spix:
            raise ValueError("Image too small")
        if spix is None:
            raise ValueError('spix needs to be specified.')
        lx = (nx//spix)*spix
        ly = (ny//spix)*spix
        rx = nx - lx
        ry = ny - ly
        out = images
        if rx:
            shiftx = np.random.randint(0, rx)
            out = out[:, shiftx:shiftx+lx]

        if ry:
            shifty = np.random.randint(0, ry)
            out = out[:,:, shifty:shifty+ly]

        if (nx//spix)>1:
            nx = out.shape[1]
            shiftx = np.random.randint(0, spix)
            new_nx = ((nx//spix)-1)*spix
            out = out[:, shiftx:shiftx+new_nx]

        if ny//spix>1:
            ny = out.shape[2]
            shifty = np.random.randint(0, spix)
            new_ny = ((ny//spix)-1)*spix
            out = out[:, :, shifty:shifty+new_ny]

    return out


def random_shift_3d(images, roll=False, spix=None, force_equal=True):
    """Apply a random shift to 3d images.

    If the roll option is not activated, the shift will not be circular and the
    data will be cropped accordingly.

    Arguments
    ---------
    roll: rolled shift (default False)
    spix: maximum size of the shift (if roll==False)
    force_equal: force all the return signals to be the same size using cropping (if roll==False)
    """
    nx = images.shape[1]
    ny = images.shape[2]
    nz = images.shape[3]
    if roll:
        shiftx = np.random.randint(0, nx)
        shifty = np.random.randint(0, ny)
        shiftz = np.random.randint(0, nz)
        out = np.roll(images, shift=shiftx, axis=1)
        out = np.roll(out, shift=shifty, axis=2)
        out = np.roll(out, shift=shiftz, axis=3)
    else:
        if spix is None:
            raise ValueError('spix needs to be specified.')
        shiftx = np.random.randint(0, spix)
        shifty = np.random.randint(0, spix)
        shiftz = np.random.randint(0, spix)
        if force_equal:
            images = images[:, :(nx//spix)*spix, :(ny//spix)*spix, :(nz//spix)*spix]
            nx = images.shape[1]
            ny = images.shape[2]
            nz = images.shape[3]

        if np.mod(nx, spix)==0:
            new_nx = ((nx//spix)-1)*spix
            out = images[:, shiftx:shiftx+new_nx, :, :]
        else:
            out = images[:, shiftx:, :, :]

        if np.mod(ny, spix)==0:
            new_ny = ((ny//spix)-1)*spix
            out = images[:, :, shifty:shifty+new_ny, :]
        else:
            out = images[:, :, shifty:, :]

        if np.mod(nz, spix)==0:
            new_nz = ((ny//spix)-1)*spix
            out = images[:, :, :, shiftz:shiftz+new_nz]
        else:
            out = images[:, :, :, shiftz:]
    return out

def random_flip_2d(images):
    ''' Apply a random flip to 2d images'''
    out = images
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=1)
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=2)
    return out


def random_transpose_2d(images):
    '''
    Apply a random transpose to 2d images
    '''
    if len(images.shape)==3:
        images = np.expand_dims(images, axis=3)
    # all possible transposes
    transposes = [(0, 1, 2, 3), (0, 2, 1, 3)]
    transpose = transposes[np.random.choice(len(transposes))]
    return np.transpose(images, axes=transpose)

def random_rotate_2d(images):
    '''
    random rotation of 2d images by multiple of 90 degree
    '''
    return random_transpose_2d(random_flip_2d(images))

def random_transformation_2d(images, *args, **kwargs):
    return random_rotate_2d(random_shift_2d(images, *args, **kwargs))



def random_flip_3d(images):
    ''' Apply a random flip to 3d images'''
    out = images
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=1)
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=2)
    if np.random.rand(1) > 0.5:
        out = np.flip(out, axis=3)
    return out


def random_transpose_3d(images):
    '''
    Apply a random transpose to 3d images
    '''

    # all possible transposes
    transposes = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
                  (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)]
    transpose = transposes[np.random.choice(len(transposes))]
    return np.transpose(images, axes=transpose)


def random_rotate_3d(images):
    '''
    random rotation of 3d images by multiple of 90 degree
    '''
    return random_transpose_3d(random_flip_3d(images))


def random_transformation_3d(images, *args, **kwargs):
    return random_rotate_3d(random_shift_3d(images,  *args, **kwargs))


def patch2img(patches, size=2):
    if size==3:
        imgs_down_left = np.concatenate([patches[:, :, :, :, 3], patches[:, :, :, :,2]], axis=2)
        imgs_down_right = np.concatenate([patches[:, :, :, :, 1], patches[:, :, :, :,0]], axis=2)
        imgs_down = np.concatenate([imgs_down_left, imgs_down_right], axis=3)
        imgs_up_left   = np.concatenate([patches[:, :, :, :, 7], patches[:, :, :, :, 6]], axis=2)
        imgs_up_right  = np.concatenate([patches[:, :, :, :, 5], patches[:, :, :, :, 4]], axis=2)
        imgs_up = np.concatenate([ imgs_up_left, imgs_up_right], axis=3)
        imgs = np.concatenate([imgs_up, imgs_down], axis=1)
    elif size==2:
        imgs_d = np.concatenate(
            [patches[:, :, :, 1], patches[:, :, :, 0]], axis=1)
        imgs_u = np.concatenate(
            [patches[:, :, :, 3], patches[:, :, :, 2]], axis=1)
        imgs = np.concatenate([imgs_u, imgs_d], axis=2)
    elif size==1:
        imgs = np.concatenate([patches[:, :, 1], patches[:, :, 0]], axis=1)
    else:
        raise ValueError('data_shape must be 1,2 or 3.')
    return imgs

def tf_patch2img_1d(r, l):
    imgs = tf.concat([l, r], axis=1)
    return imgs

def tf_patch2img_2d(dr, dl, ur, ul):

    imgs_d = tf.concat([dl, dr], axis=1)
    imgs_u = tf.concat([ul, ur], axis=1)
    imgs = tf.concat([imgs_u,  imgs_d], axis=2)
    return imgs

def tf_patch2img_3d(*args):
    imgs_down_left = tf.concat([args[3], args[2]], axis=2)
    imgs_down_right = tf.concat([args[1], args[0]], axis=2)
    imgs_down = tf.concat([imgs_down_left, imgs_down_right], axis=3)
    imgs_up_left = tf.concat([args[7], args[6]], axis=2)
    imgs_up_right = tf.concat([args[5], args[4]], axis=2)
    imgs_up = tf.concat([ imgs_up_left, imgs_up_right], axis=3)
    imgs = tf.concat([imgs_up, imgs_down], axis=1)
    return imgs

def flip_slices_1d(l):
    flip_l = np.flip(l, axis=1)
    return flip_l

def tf_flip_slices_1d(l):
    return tf.reverse(l, axis=[1])

def flip_slices_2d(dl, ur, ul):
    flip_dl = np.flip(dl, axis=1)
    flip_ur = np.flip(ur, axis=2)    
    flip_ul = np.flip(np.flip(ul, axis=1), axis=2)
    return flip_dl, flip_ur, flip_ul

def tf_flip_slices_2d(dl, ur, ul):
    flip_dl = tf.reverse(dl, axis=[1])
    flip_ur = tf.reverse(ur, axis=[2])    
    flip_ul = tf.reverse(ul, axis=[1,2])
    return flip_dl, flip_ur, flip_ul

def flip_slices_3d(*args):
    flip_d_above = np.flip(args[0], axis=2)
    flip_d_left = np.flip(args[1], axis=3)
    flip_d_corner = np.flip(np.flip(args[2], axis=2), axis=3)
    flip_up = np.flip(args[3], axis=1)
    flip_u_above = np.flip(np.flip(args[4], axis=1), axis=2)
    flip_u_left = np.flip(np.flip(args[5], axis=1), axis=3)
    flip_u_corner = np.flip(np.flip(np.flip(args[6], axis=1), axis=2), axis=3)
    return flip_d_above, flip_d_left, flip_d_corner, flip_up, flip_u_above, flip_u_left, flip_u_corner

def tf_flip_slices_3d(*args):
    flip_d_above = tf.reverse(args[0], axis=[2])
    flip_d_left = tf.reverse(args[1], axis=[3])
    flip_d_corner = tf.reverse(args[2], axis=[2, 3])
    flip_up = tf.reverse(args[3], axis=[1])
    flip_u_above = tf.reverse(args[4], axis=[1, 2])
    flip_u_left = tf.reverse(args[5], axis=[1, 3])
    flip_u_corner = tf.reverse(args[6], axis=[1, 2, 3])
    return flip_d_above, flip_d_left, flip_d_corner, flip_up, flip_u_above, flip_u_left, flip_u_corner

def tf_flip_slices(*args, size=2):
    if size==3:
        return tf_flip_slices_3d(*args)
    elif size==2:
        return tf_flip_slices_2d(*args)
    elif size==1:
        return tf_flip_slices_1d(*args)        
    else:
        raise ValueError("Size should be 1, 2 or 3")


def tf_patch2img(*args, size=2):
    if size==3:
        return tf_patch2img_3d(*args)
    elif size==2:
        return tf_patch2img_2d(*args)
    elif size==1:
        return tf_patch2img_1d(*args)
    else:
        raise ValueError("Size should be 1, 2 or 3")


def slice_time(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    s = cubes.shape
    # The last dimension is used for the samples
    cubes = cubes.transpose([1, 2, 3, 0])

    # compute the number of slices (We assume square images)
    num_slices = cubes.shape[1] // spix

    # To ensure left over pixels in each dimension are ignored
    limit = num_slices * spix
    cubes = cubes[:, :limit, :limit]

    # split along first dimension
    sliced_dim1 = np.vstack(np.split(cubes, num_slices, axis=1))

    # split along second dimension
    sliced_dim2 = np.vstack(np.split(sliced_dim1, num_slices, axis=2))

    return sliced_dim2

def slice_1d(signal, spix=256):
    '''
    slice the signal
    '''
    s = signal.shape
    if len(signal.shape)==1:
        signal = np.reshape(signal, [1,len(signal)])
    
    # compute the number of slices (We assume square images)
    num_slices = signal.shape[1] // spix

    # To ensure left over pixels in each dimension are ignored
    limit = num_slices * spix
    cubes = signal[:, :limit]

    # split along first dimension
    sliced_signal = np.vstack(np.split(cubes, num_slices, axis=1))

    return sliced_signal


def slice_2d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    s = cubes.shape


    # This was done in a old version of the code
    # # The last dimension is used for the samples
    # cubes = cubes.transpose([0, 3, 1, 2])
    # cubes = cubes.reshape([s[0] * s[3], s[1], s[2]])

    # compute the number of slices (We assume square images)
    num_slices_x = s[1] // spix
    num_slices_y = s[2] // spix

    # To ensure left over pixels in each dimension are ignored
    limit_x = num_slices_x * spix
    limit_y = num_slices_y * spix
    out = cubes[:, :limit_x, :limit_y]

    if num_slices_x>1:
        # split along first dimension
        out = np.vstack(np.split(out, num_slices_x, axis=1))

    if num_slices_y>1:
        # split along second dimension
        out = np.vstack(np.split(out, num_slices_y, axis=2))

    return out


def slice_3d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''
    num_slices_dim_1 = cubes.shape[1] // spix
    num_slices_dim_2 = cubes.shape[2] // spix
    num_slices_dim_3 = cubes.shape[3] // spix

    # To ensure left over pixels in each dimension are ignored
    limit_dim_1 = num_slices_dim_1 * spix
    limit_dim_2 = num_slices_dim_2 * spix
    limit_dim_3 = num_slices_dim_3 * spix

    cubes = cubes[:, :limit_dim_1, :limit_dim_2, :limit_dim_3]

    # split along first dimension
    cubes = np.vstack(np.split(cubes, num_slices_dim_1, axis=1))
    # split along second dimension
    cubes = np.vstack(np.split(cubes, num_slices_dim_2, axis=2))
    # split along third dimension
    cubes = np.vstack(np.split(cubes, num_slices_dim_3, axis=3))

    return cubes


def slice_1d_patch(img0, spix=64):

    # Handle the dimesnsions
    l = len(img0.shape)
    if l < 1:
        ValueError('Not enough dimensions')
    elif l == 1:
        img0 = img0.reshape([1, *img0.shape, 1])
    elif l == 2:
        img0 = img0.reshape([*img0.shape, 1])
    elif l > 3:
        ValueError('To many dimensions')
    sx= img0.shape[1]

    nx = sx // spix
    img0 = img0[:,:nx*spix]

    # 1) Create the different subparts
    img1 = np.roll(img0[:,:,0:1], spix, axis=1)
    img1[:, :spix] = 0

    # 2) Concatenate
    img = np.concatenate([img0, img1], axis=2)

    # 3) Slice the image
    img = np.squeeze(np.vstack(np.split(img, nx, axis=1)))

    return img

def slice_2d_patch(img0, spix=64):

    # Handle the dimesnsions
    l = len(img0.shape)
    if l < 2:
        ValueError('Not enough dimensions')
    elif l == 2:
        img0 = img0.reshape([1, *img0.shape, 1])
    elif l == 3:
        img0 = np.expand_dims(img0, axis=3)
    elif l > 4:
        ValueError('To many dimensions')
    _, sx, sy = img0.shape[:3]
    nx = sx // spix
    ny = sy // spix

    # 1) Create the different subparts
    img1 = np.roll(img0, spix, axis=1)
    img1[:, :spix, :] = 0

    img2 = np.roll(img0, spix, axis=2)
    img2[:, :, :spix] = 0

    img3 = np.roll(img1, spix, axis=2)
    img3[:, :, :spix] = 0

    # 2) Concatenate
    img = np.concatenate([img0, img1, img2, img3], axis=3)
    
    del img1, img2, img3
    # 3) Slice the image
    if nx>1:
        img = np.vstack(np.split(img, nx, axis=1))
    if ny>1:
        img = np.vstack(np.split(img, ny, axis=2))

    return img


def slice_3d_patch(cubes, spix=32):
    '''
    cubes: the 3d histograms - [:, :, :, :]
    '''

    # Handle the dimesnsions
    l = len(cubes.shape)
    if l < 3:
        ValueError('Not enough dimensions')
    elif l == 3:
        cubes = cubes.reshape([1, *cubes.shape]) # add one extra dimension for number of cubes
    elif l > 4:
        ValueError('To many dimensions')

    _, sx, sy, sz = cubes.shape
    nx = sx // spix
    ny = sy // spix
    nz = sz // spix

    # 1) Create all 7 neighbors for each smaller cube
    img1 = np.roll(cubes, spix, axis=2)
    img1[:, :, :spix, :] = 0

    img2 = np.roll(cubes, spix, axis=3)
    img2[:, :, :, :spix] = 0
    
    img3 = np.roll(img1, spix, axis=3)
    img3[:, :, :, :spix] = 0
    
    img4 = np.roll(cubes, spix, axis=1) # extra for the 3D case
    img4[:, :spix, :, :] = 0
    
    img5 = np.roll(img4, spix, axis=2)
    img5[:, :, :spix, :] = 0
    
    img6 = np.roll(img4, spix, axis=3)
    img6[:, :, :, :spix] = 0
    
    img7 = np.roll(img5, spix, axis=3)
    img7[:, :, :, :spix] = 0
    

    # 2) Concatenate
    img_with_nbrs = np.stack([cubes, img1, img2, img3, img4, img5, img6, img7], axis=4) # 7 neighbors plus the original cube
    
    # Clear variable to gain some RAM
    del img1, img2, img3, img4, img5, img6, img7


    # 3) Slice the cubes
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nx, axis=1))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, ny, axis=2))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nz, axis=3))

    return img_with_nbrs


# This function is in testing mode...
def downsample_1d(sig, s=2, Nwin=32):
    if len(sig.shape)==2:
        return np.apply_along_axis(downsample_1d,1, sig, s=s, Nwin=Nwin)
    win = firwin(numtaps=Nwin, cutoff=1/2)
    ntimes = np.log2(s)
    assert(ntimes-np.int(ntimes)<1e-6)
    ntimes = np.int(np.round(ntimes))
    new_sig = sig.copy()
    for _ in range(ntimes):
        new_sig = np.convolve(new_sig,win, 'same')
        new_sig = new_sig[1::2]
    return new_sig

# This function is in testing mode...
def upsamler_1d(sig, s=2, Nwin=32):
    if len(sig.shape)==2:
        return np.apply_along_axis(upsamler_1d, 1, sig, s=s, Nwin=Nwin)
    win = firwin(numtaps=Nwin, cutoff=4/7)
    ntimes = np.log2(s)
    assert(ntimes-np.int(ntimes)<1e-6)
    ntimes = np.int(np.round(ntimes))
    tsig = sig.copy()
    for _ in range(ntimes):
        new_sig = np.zeros(shape=[len(tsig)*2])
        new_sig[1::2] = tsig
        new_sig[::2] = tsig
        new_sig = np.convolve(new_sig,win, 'same')
        tsig = new_sig
    return new_sig
