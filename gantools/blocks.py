import tensorflow as tf
import numpy as np
from numpy import prod
from gantools import utils


def orthogonal_initializer(scale=1.1):
    """From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    """

    def _initializer(shape, dtype=tf.float32, **kwargs):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)  #this needs to be corrected to float32
        # print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)

    return _initializer


def select_initializer(type='xavier'):
    if type=='orthogonal':
        return orthogonal_initializer()
    elif type=='xavier':
        return tf.contrib.layers.xavier_initializer()
    else:
        raise ValueError('Unknown initializer type.')
    return

def _tf_variable(name, shape, initializer):
    """Create a tensorflow variable.

    Arguments
    --------
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    if True:  # with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def numel(x):
    ''' Return the number of element in x '''
    return prod(tf.shape(x))  # tf.size?


def reshape2d(x, name=None):
    """Squeeze x into a 2d matrix."""
    return tf.reshape(
        x, [tf.shape(x)[0], prod(x.shape.as_list()[1:])], name=name)


def reshape4d(x, sx, sy, nc, name=None):
    """Squeeze x into a 2d matrix."""
    return tf.reshape(x, [tf.shape(x)[0], sx, sy, nc], name=name)


def lrelu(x, leak=0.2, name="lrelu"):
    """Leak relu."""
    return tf.maximum(x, leak * x, name=name)


def selu(x, name="selu"):
    return tf.nn.selu(x, name=name)


def prelu(x, name="prelu"):
    """Parametrized Rectified Linear Unit, He et al. 2015, https://arxiv.org/abs/1502.01852"""
    return tf.keras.layers.PReLU(name=name)(x)


def batch_norm(x, epsilon=1e-5, momentum=0.9, name="batch_norm", train=True):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(
            x,
            decay=momentum,
            updates_collections=None,
            epsilon=epsilon,
            scale=True,
            is_training=train,
            scope=name)

        return bn


def np_downsample_1d(x, scaling):
    unique = False
    if len(x.shape) < 1:
        raise ValueError('Too few dimensions')
    elif len(x.shape) == 1:
        x = x.reshape([1, *x.shape, 1])
        unique = True
        unichannel = True
    elif len(x.shape) == 2:
        x = x.reshape([*x.shape, 1])
        unichannel = True
    elif len(x.shape) > 3:
        raise ValueError('Too many dimensions')

    n, sx, c = x.shape
    nx = sx // scaling
    dsx = np.zeros([n, nx, c])
    for i in range(scaling):
        dsx += x[:, i:scaling * nx + i:scaling]
    dsx /= scaling
    if unichannel:
        dsx = dsx[:, :, 0]
    if unique:
        dsx = dsx[0]
    return dsx


def np_downsample_2d(x, scaling):
    unique = False
    unichannel = False
    if len(x.shape) < 2:
        raise ValueError('Too few dimensions')
    elif len(x.shape) == 2:
        x = x.reshape([1, *x.shape, 1])
        unique = True
        unichannel = True
    elif len(x.shape) == 3:
        x = x.reshape([*x.shape, 1])
        unichannel = True
    elif len(x.shape) > 4:
        raise ValueError('Too many dimensions')
    n, sx, sy, c = x.shape
    nx = sx // scaling
    ny = sy // scaling
    dsx = np.zeros([n, nx, ny, c])
    for i in range(scaling):
        for j in range(scaling):
            dsx += x[:, i:scaling * nx + i:scaling, j:scaling * ny +
                     j:scaling]
    dsx /= (scaling**2)
    if unichannel:
        dsx = dsx[:, :, :, 0]
    if unique:
        dsx = dsx[0]
    return dsx


def np_downsample_3d(x, scaling):
    unique = False
    unichannel = False
    if len(x.shape) < 3:
        raise ValueError('Too few dimensions')
    elif len(x.shape) == 3:
        x = x.reshape([1, *x.shape, 1])
        unique = True
        unichannel = True
    elif len(x.shape) == 4:
        x = x.reshape([*x.shape, 1])
        unichannel = True
    elif len(x.shape) > 5:
        raise ValueError('Too many dimensions')

    n, sx, sy, sz, c = x.shape
    nx = sx // scaling
    ny = sy // scaling
    nz = sz // scaling
    dsx = np.zeros([n, nx, ny, nz, c])
    for i in range(scaling):
        for j in range(scaling):
            for k in range(scaling):
                dsx += x[:,i:scaling*nx+i:scaling,j:scaling*ny+j:scaling,k:scaling*nz+k:scaling]
    dsx /= (scaling**3)
    if unichannel:
        dsx = dsx[:, :, :, :, 0]
    if unique:
        dsx = dsx[0]
    return dsx

def downsample(imgs, s, size=None):
    if size is None:
        size = utils.get_data_size(imgs)
    if size == 3:
        return np_downsample_3d(imgs, s)
    elif size == 2:
        return np_downsample_2d(imgs, s)
    elif size == 1:
        return np_downsample_1d(imgs, s)
    else:
        raise ValueError("Size should be 1,2,3 or None")



def down_sampler(x=None, s=2, size=None):
    '''
    Op to downsample 2D or 3D images by factor 's'.
    This method works for both inputs: tensor or placeholder
    '''

    if size is None:
        size = utils.get_data_size(x)

    if not (size in [1, 2, 3]):
        raise ValueError("Wrong size parameter")

    # The input to the downsampling operation is a placeholder.
    if x is None:
        if size == 3:
            addname = '3d_'
        elif size == 2:
            addname = '2d_'
        else:
            addname = '1d_'
        placeholder_name = 'down_sampler_in_' + addname + str(s)
        down_sampler_x = tf.placeholder(dtype=tf.float32, name=placeholder_name)
        op_name = 'down_sampler_out_' + addname + str(s)
    
    # The input to the downsampling operation is the input tensor x.
    else: 
        down_sampler_x = x
        op_name = None

    if size==3:
        filt = tf.constant(1 / (s * s * s), dtype=tf.float32, shape=[s, s, s, 1, 1])
        return tf.nn.conv3d(down_sampler_x, filt, strides=[1, s, s, s, 1], padding='SAME', name=op_name)

    elif size==2:
        filt = tf.constant(1 / (s * s), dtype=tf.float32, shape=[s, s, 1, 1])
        if down_sampler_x.shape[-1]==1:
            return tf.nn.conv2d(down_sampler_x, filt, strides=[1, s, s, 1], padding='SAME', name=op_name)
        else:
            res = []
            for sl in tf.split(down_sampler_x, down_sampler_x.shape[-1], axis=3):
                if op_name is not None:
                    op_name += 'I'
                res.append(tf.nn.conv2d(sl, filt, strides=[1, s, s, 1], padding='SAME', name=op_name))
            return tf.concat(res, axis=3)
    else:
        filt = tf.constant(1 / s, dtype=tf.float32, shape=[s, 1, 1])
        return tf.nn.conv1d(down_sampler_x, filt, stride=s, padding='SAME', name=op_name)

def up_sampler(x, s=2, size=None, smoothout=False):
    if size is None:
        size = utils.get_data_size(x)

    if not (size in [1, 2, 3]):
        raise ValueError("Wrong size parameter")

    bs = tf.shape(x)[0]
    dims = x.shape.as_list()[1:]

    if size == 3:
        filt = tf.constant(1, dtype=tf.float32, shape=[s, s, s, 1, 1])
        output_shape = [bs, dims[0] * s, dims[1] * s, dims[2] * s, dims[3]]
        return tf.nn.conv3d_transpose(
            x,
            filt,
            output_shape=output_shape,
            strides=[1, s, s, s, 1],
            padding='SAME')
    elif size == 2:
        filt = tf.constant(1, dtype=tf.float32, shape=[s, s, 1, 1])
        output_shape = [bs, dims[0] * s, dims[1] * s, 1]
        if dims[-1]==1:
            x = tf.nn.conv2d_transpose(
                x,
                filt,
                output_shape=output_shape,
                strides=[1, s, s, 1],
                padding='SAME')
            if smoothout:
                paddings = tf.constant([[0,0],[s//2-1, s//2-1], [ s//2,  s//2], [0,0]])
                x = tf.pad(x, paddings, "SYMMETRIC")
                x = tf.nn.conv2d(
                    x,
                    filt/(s*s),
                    strides=[1, 1, 1, 1],
                    padding='VALID')
            return x
        else:
            res = []
            for sl in tf.split(x, dims[-1], axis=3):
                tx = tf.nn.conv2d_transpose(
                    sl,
                    filt,
                    output_shape=output_shape,
                    strides=[1, s, s, 1],
                    padding='SAME')
                if smoothout:
                    paddings = tf.constant([[0,0],[s//2-1, s//2], [ s//2-1,  s//2], [0,0]])
                    tx = tf.pad(tx, paddings, "SYMMETRIC")
                    tx = tf.nn.conv2d(
                        tx,
                        filt/(s*s),
                        strides=[1, 1, 1, 1],
                        padding='VALID')
                res.append(tx)
            return tf.concat(res, axis=3)
    else:
        filt = tf.constant(1, dtype=tf.float32, shape=[s, 1, 1])
        output_shape = [bs, dims[0] * s, dims[1]]
        return tf.contrib.nn.conv1d_transpose(
            x,
            filt,
            output_shape=output_shape,
            stride= s,
            padding='SAME')



def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm



def conv1d(imgs, nf_out, shape=[5], stride=2, use_spectral_norm=False, name="conv1d", summary=True):
    '''Convolutional layer for square images'''
    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride]
    weights_initializer = select_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _tf_variable(
            'w', [shape[0],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
        conv = tf.nn.conv1d(
            imgs, w, stride=stride[0], padding='SAME')

        biases = _tf_variable('biases', [nf_out], initializer=const)
        conv = tf.nn.bias_add(conv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv

def conv2d(imgs, nf_out, shape=[5, 5], stride=2, use_spectral_norm=False, name="conv2d", summary=True):
    '''Convolutional layer for square images'''

    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride, stride]

    weights_initializer = select_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _tf_variable(
            'w', [shape[0], shape[1],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
        conv = tf.nn.conv2d(
            imgs, w, strides=[1, *stride, 1], padding='SAME')

        biases = _tf_variable('biases', [nf_out], initializer=const)
        conv = tf.nn.bias_add(conv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv


def conv3d(imgs,
           nf_out,
           shape=[5, 5, 5],
           stride=2,
           use_spectral_norm=False,
           name="conv3d",
           summary=True):
    '''Convolutional layer for square images'''
    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride, stride, stride]
    if use_spectral_norm:
        print("Warning spectral norm for conv3d set to True but may not be implemented!")

    weights_initializer = select_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        w = _tf_variable(
            'w', [shape[0], shape[1], shape[2],
                  imgs.get_shape()[-1], nf_out],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
        conv = tf.nn.conv3d(
            imgs, w, strides=[1, *stride, 1], padding='SAME')

        biases = _tf_variable('biases', [nf_out], initializer=const)
        conv = tf.nn.bias_add(conv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])

        return conv

def deconv1d(imgs,
             output_shape,
             shape=[5],
             stride=2,
             name="deconv1d",
             use_spectral_norm=False,
             summary=True):
    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride]
    weights_initializer = select_initializer()
    # was
    # weights_initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = _tf_variable(
            'w', [shape[0], output_shape[-1],
                  imgs.get_shape()[-1]],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
        deconv = tf.contrib.nn.conv1d_transpose(
            imgs,
            w,
            output_shape=output_shape,
            stride=stride[0])

        biases = _tf_variable(
            'biases', [output_shape[-1]], initializer=const)
        deconv = tf.nn.bias_add(deconv, biases)

        # If we are running on Leonhard we need to reshape in order for TF
        # to explicitly know the shape of the tensor. Machines with newer
        # TensorFlow versions do not need this.
        if tf.__version__ == '1.3.0':
            deconv = tf.reshape(deconv, output_shape)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])
        return deconv

def deconv2d(imgs,
             output_shape,
             shape=[5, 5],
             stride=2,
             name="deconv2d",
             use_spectral_norm=False,
             summary=True):
    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride, stride]
    weights_initializer = select_initializer()
    # was
    # weights_initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = _tf_variable(
            'w', [shape[0], shape[1], output_shape[-1],
                  imgs.get_shape()[-1]],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
        deconv = tf.nn.conv2d_transpose(
            imgs,
            w,
            output_shape=output_shape,
            strides=[1, *stride, 1])

        biases = _tf_variable(
            'biases', [output_shape[-1]], initializer=const)
        deconv = tf.nn.bias_add(deconv, biases)

        # If we are running on Leonhard we need to reshape in order for TF
        # to explicitly know the shape of the tensor. Machines with newer
        # TensorFlow versions do not need this.
        if tf.__version__ == '1.3.0':
            deconv = tf.reshape(deconv, output_shape)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])
        return deconv


def deconv3d(imgs,
             output_shape,
             shape=[5, 5, 5],
             stride=2,
             name="deconv3d",
             use_spectral_norm=False,
             summary=True):
    if not(isinstance(stride ,list) or isinstance(stride ,tuple)):
        stride = [stride, stride, stride]
    weights_initializer = select_initializer()
    # was
    # weights_initializer = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        # filter : [depth, height, width, output_channels, in_channels]
        w = _tf_variable(
            'w', [
                shape[0], shape[1], shape[2], output_shape[-1],
                imgs.get_shape()[-1]
            ],
            initializer=weights_initializer)
        if use_spectral_norm:
            w = spectral_norm(w)
        deconv = tf.nn.conv3d_transpose(
            imgs,
            w,
            output_shape=output_shape,
            strides=[1, *stride, 1])

        biases = _tf_variable(
            'biases', [output_shape[-1]],
            initializer=const)  # one bias for each filter
        deconv = tf.nn.bias_add(deconv, biases)

        if summary:
            tf.summary.histogram("Bias_sum", biases, collections=["metrics"])
            # we put it in metrics so we don't store it too often
            tf.summary.histogram("Weights_sum", w, collections=["metrics"])
        return deconv


def inception_deconv(in_tensor, bs, sx, n_filters, stride, summary, num, data_size=2, use_spectral_norm=False, merge=False):
    if data_size == 3:
        output_shape = [bs, sx, sx, sx, n_filters]
        deconv = deconv3d
        shape = [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
        concat_axis = 4
    elif data_size == 2:
        output_shape = [bs, sx, sx, n_filters]
        deconv = deconv2d
        shape = [[1, 1], [3, 3], [5, 5]]
        concat_axis = 3
    else:
        raise NotImplementedError

    tensor_1 = deconv(in_tensor,
                          output_shape=output_shape,
                          shape=shape[0],
                          stride=stride,
                          name='{}_deconv_1_by_1'.format(num),
                          use_spectral_norm=use_spectral_norm,
                          summary=summary)

    tensor_3 = deconv(in_tensor,
                          output_shape=output_shape,
                          shape=shape[1],
                          stride=stride,
                          name='{}_deconv_3_by_3'.format(num),
                          use_spectral_norm=use_spectral_norm,

                          summary=summary)

    tensor_5 = deconv(in_tensor,
                          output_shape=output_shape,
                          shape=shape[2],
                          stride=stride,
                          name='{}_deconv_5_by_5'.format(num),
                          use_spectral_norm=use_spectral_norm,
                          summary=summary)

    out_tensor = tf.concat([tensor_1, tensor_3, tensor_5], axis=concat_axis)

    if merge:
        # do 1x1 convolution to reduce the number of output channels from (3 x n_filters) to n_filters
        out_tensor = deconv(out_tensor,
                          output_shape=output_shape,
                          shape=shape[0],
                          stride=1,
                          name='{}_deconv_1_by_1_merge'.format(num),
                          use_spectral_norm=use_spectral_norm,
                          summary=summary)

    return out_tensor

def inception_conv(in_tensor, n_filters, stride, summary, num, data_size=2, use_spectral_norm=False, merge=False):
    if data_size == 3:
        conv = conv3d
        shape = [[1, 1, 1], [3, 3, 3], [5, 5, 5]]
        concat_axis = 4
    elif data_size == 2:
        conv = conv2d
        shape = [[1, 1], [3, 3], [5, 5]]
        concat_axis = 3
    else:
        raise NotImplementedError

    tensor_1 = conv(in_tensor,
                    nf_out=n_filters,
                    shape=shape[0],
                    stride=stride,
                    name='{}_conv_1_by_1'.format(num),
                    use_spectral_norm=use_spectral_norm,
                    summary=summary)

    tensor_3 = conv(in_tensor,
                    nf_out=n_filters,
                    shape=shape[1],
                    stride=stride,
                    name='{}_conv_3_by_3'.format(num),
                    use_spectral_norm=use_spectral_norm,
                    summary=summary)

    tensor_5 = conv(in_tensor,
                 nf_out=n_filters,
                 shape=shape[2],
                 stride=stride,
                 name='{}_conv_5_by_5'.format(num),
                 use_spectral_norm=use_spectral_norm,
                 summary=summary)

    out_tensor = tf.concat([tensor_1, tensor_3, tensor_5], axis=concat_axis)

    if merge:
        # do 1x1 convolution to reduce the number of output channels from (3 x n_filters) to n_filters
        out_tensor = conv(out_tensor,
                        nf_out=n_filters,
                        shape=shape[0],
                        stride=1,
                        name='{}_conv_1_by_1_merge'.format(num),
                        use_spectral_norm=use_spectral_norm,
                        summary=summary)

    return out_tensor


def linear(input_, output_size, scope=None, summary=True):
    shape = input_.get_shape().as_list()

    weights_initializer = select_initializer()
    const = tf.constant_initializer(0.0)

    with tf.variable_scope(scope or "Linear"):
        matrix = _tf_variable(
            "Matrix", [shape[1], output_size],
            initializer=weights_initializer)
        bias = _tf_variable("bias", [output_size], initializer=const)
        if summary:
            tf.summary.histogram(
                "Matrix_sum", matrix, collections=["metrics"])
            tf.summary.histogram("Bias_sum", bias, collections=["metrics"])
        return tf.matmul(input_, matrix) + bias


def mini_batch_reg(xin, n_kernels=300, dim_per_kernel=50):
    x = linear(xin, n_kernels * dim_per_kernel, scope="minibatch_reg")
    activation = tf.reshape(x, [tf.shape(x)[0], n_kernels, dim_per_kernel])
    abs_dif = tf.reduce_sum(
        tf.abs(
            tf.expand_dims(activation, 3) -
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
    C = tf.exp(-abs_dif)
    minibatch_features = (tf.reduce_sum(C, 2) - 1) / (
        tf.subtract(tf.cast(tf.shape(x)[0], tf.float32), 1.0))
    x = tf.concat([xin, minibatch_features], axis=1)

    return x

def apply_phaseshuffle(x, rad=2, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

def tf_cdf(x, n_out, name='cdf_weight', diff_weight=10, use_first_channel=True):

    """Helping function to get correct histograms."""
    # limit = 4.
    # wi = tf.range(0.0, limit, delta=limit/n_out, dtype=tf.float32, name='range')

    # wl = tf.Variable(
    #     tf.reshape(wi, shape=[1, 1, n_out]),
    #     name='cdf_weight_left',
    #     dtype=tf.float32)
    # wr = tf.Variable(
    #     tf.reshape(wi, shape=[1, 1, n_out]),
    #     name='cdf_weight_right',
    #     dtype=tf.float32)
    weights_initializer = select_initializer()
    wr = _tf_variable(
        name+'_right',
        shape=[1, 1, n_out],
        initializer=weights_initializer)
    wl = _tf_variable(
        name+'_left',
        shape=[1, 1, n_out],
        initializer=weights_initializer)
    if use_first_channel:
        nc = len(x.shape)
        if nc == 4:
            x = x[:, :, :, 0]
        elif nc == 5:
            x = x[:, :, :, :, 0]
        else:
            raise ValueError('Wrong size')
    x = tf.expand_dims(reshape2d(x), axis=2)
    xl = tf.reduce_mean(tf.sigmoid(10 * (wl - x)), axis=1)
    xr = tf.reduce_mean(tf.sigmoid(10 * (x - wr)), axis=1)
    tf.summary.histogram("cdf_weight_right", wr, collections=["metrics"])
    tf.summary.histogram("cdf_weight_left", wl, collections=["metrics"])

    return tf.concat([xl, xr], axis=1)


def tf_covmat(x, shape):
    if x.shape[-1] > 1:
        lst = []
        for i in range(x.shape[-1]):
            lst.append(tf_covmat(x[:,:,:,i:i+1], shape))
        return tf.stack(lst, axis=1)
    nel = np.prod(shape)
    bs = tf.shape(x)[0]

    sh = [shape[0], shape[1], 1, nel]
    # wi = tf.constant_initializer(0.0)
    # w = _tf_variable('covmat_var', sh, wi)
    w = tf.constant(np.eye(nel).reshape(sh), dtype=tf.float32)

    conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
    nx = conv.shape[1]*conv.shape[2]
    conv_vec = tf.reshape(conv,shape=[bs, nx ,nel])
    m = tf.reduce_mean(conv_vec, axis=[1,2])
    conv_vec = tf.subtract(conv_vec,tf.expand_dims(tf.expand_dims(m,axis=1), axis=2))
    c = 1/tf.cast(nx, tf.float32)*tf.matmul(tf.transpose(conv_vec,perm=[0,2,1]), conv_vec)
    return c


# def learned_histogram(x, params):
#     """A learned histogram layer.

#     The center and width of each bin is optimized.
#     One histogram is learned per feature map.
#     """
#     # Shape of x: #samples x #nodes x #features.
#     bins = params.get('bins', 20)
#     initial_range = params.get('initial_range', 2)
#     data_size = params.get('data_size', 2)

#         w = _tf_variable(
#             'w', [shape[0], shape[1],
#                   imgs.get_shape()[-1], nf_out],
#             initializer=weights_initializer)
#         conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

#         biases = _tf_variable('biases', [nf_out], initializer=const)
#         conv = tf.nn.bias_add(conv, biases)
#     centers = tf.linspace(float(0), initial_range, bins, name='range')
#     centers = tf.expand_dims(centers, axis=1)
#     centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
#     centers = tf.Variable(
#         tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
#         name='centers', dtype=tf.float32)
#     # width = bins * initial_range / 4  # 50% overlap between bins.
#     width = 1 / initial_range  # 50% overlap between bins.
#     widths = tf.get_variable(
#         name='widths', shape=[1, 1, n_features, bins], dtype=tf.float32,
#         initializer=tf.initializers.constant(value=width, dtype=tf.float32))
#     x = tf.expand_dims(x, axis=3)
#     # All are rank-4 tensors: samples, nodes, features, bins.
#     widths = tf.abs(widths)
#     dist = tf.abs(x - centers)
#     hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1)
#     return tf.reshape(hist, [tf.shape(hist)[0], hist.shape[1] * hist.shape[2]])

def learned_histogram(x, params):
    """A learned histogram layer.

    The center and width of each bin is optimized.
    One histogram is learned per feature map.
    """
    # Shape of x: #samples x #nodes x #features.
    bins = params.get('bins', 20)
    initial_range = params.get('initial_range', 2)
    is_3d = params.get('is_3d', False)
    if is_3d:
        x = tf.reshape(x, [tf.shape(x)[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4]])
    else:
        x = tf.reshape(x, [tf.shape(x)[0], x.shape[1] * x.shape[2], x.shape[3]])
    n_features = int(x.get_shape()[2])
    centers = tf.linspace(float(0), initial_range, bins, name='range')
    centers = tf.expand_dims(centers, axis=1)
    centers = tf.tile(centers, [1, n_features])  # One histogram per feature channel.
    centers = tf.Variable(
        tf.reshape(tf.transpose(centers), shape=[1, 1, n_features, bins]),
        name='centers', dtype=tf.float32)
    # width = bins * initial_range / 4  # 50% overlap between bins.
    width = 1 / initial_range  # 50% overlap between bins.
    widths = tf.get_variable(
        name='widths', shape=[1, 1, n_features, bins], dtype=tf.float32,
        initializer=tf.initializers.constant(value=width, dtype=tf.float32))
    x = tf.expand_dims(x, axis=3)
    # All are rank-4 tensors: samples, nodes, features, bins.
    widths = tf.abs(widths)
    dist = tf.abs(x - centers)
    hist = tf.reduce_mean(tf.nn.relu(1 - dist * widths), axis=1)
    return tf.reshape(hist, [tf.shape(hist)[0], hist.shape[1] * hist.shape[2]])
