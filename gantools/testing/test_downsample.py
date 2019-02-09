if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))
    os.environ["CUDA_VISIBLE_DEVICES"]=""

import unittest

import numpy as np
import tensorflow as tf
    
from gantools import blocks

def downsample_old(imgs, s, is_3d=False, sess=None):
    '''
    Makes sure that multiple nodes are not created for the same downsampling op.
    If the downsampling node does not exist in the graph, create. If exists, then reuse it.
    '''
    if sess is None:
        new_sess = tf.Session()
    else:
        new_sess = sess

    # name of the ouput tensor after the downsampling operation
    down_sampler_out_name = 'down_sampler_out_' + ('3d_' if is_3d else '2d_') + str(s) + ':0'

    # Don't create a node for the op if one already exists in the computation graph with the same name
    try:
        down_sampler_op = tf.get_default_graph().get_tensor_by_name(down_sampler_out_name)

    except KeyError as e:
        print('Tensor {} not found, hence creating the Op.'.format(down_sampler_out_name))
        if is_3d:
            size = 3
        else:
            size = 2
        down_sampler_op = blocks.down_sampler(x=None, s=s, size=size)


    # name of the input placeholder to the downsapling operation
    placeholder_name = 'down_sampler_in_' + ('3d_' if is_3d else '2d_') + str(s) + ':0'
    placeholder = tf.get_default_graph().get_tensor_by_name(placeholder_name)


    if is_3d:
        # 1 extra dim for channels
        if len(imgs.shape) < 5:
            imgs = np.expand_dims(imgs, axis=4)

        img_d = new_sess.run(down_sampler_op, feed_dict={placeholder : imgs})
        ret = np.squeeze(img_d)

    else:
        if len(imgs.shape) < 4:
            imgs = np.expand_dims(imgs, axis=3)
        
        img_d = []
        for i in range(imgs.shape[3]):
            curr_img = np.expand_dims(imgs[:, :, :, i], axis=3)
            img_d.append(new_sess.run(down_sampler_op, feed_dict={placeholder: curr_img}))

        ret = np.squeeze(np.concatenate(img_d, axis=3))

    # If a new session was created, close it. 
    if sess is None:
        new_sess.close()

    return ret

class TestDownsample(unittest.TestCase):
    def test_2d(self):
        tmp_data = np.random.rand(16,16,64)
        d1 = downsample_old(tmp_data, 2, False)
        d2 = blocks.downsample(tmp_data, 2, 2)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

        tmp_data = np.random.randn(16,16,64)
        d1 = downsample_old(tmp_data, 4, False)
        d2 = blocks.downsample(tmp_data, 4, 2)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

        tmp_data = np.random.rand(16,16,64)
        d1 = downsample_old(tmp_data, 8, False)
        d2 = blocks.downsample(tmp_data, 8, 2)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

        tmp_data =  np.random.randint(0,255,size=[16,16,64])
        d1 = downsample_old(tmp_data, 8, False)
        d2 = blocks.downsample(tmp_data, 8, 2)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

    def test_3d(self):
        tmp_data = np.random.rand(16,16,8,64)
        d1 = downsample_old(tmp_data, 2, True)
        d2 = blocks.downsample(tmp_data, 2, 3)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

        tmp_data = np.random.randn(16,16,8,64)
        d1 = downsample_old(tmp_data, 4, True)
        d2 = blocks.downsample(tmp_data, 4, 3)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

        tmp_data = np.random.randn(16,16,32,64)
        d1 = downsample_old(tmp_data, 8, True)
        d2 = blocks.downsample(tmp_data, 8, 3)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

        tmp_data =  np.random.randint(0,255,size=[16,16,64,32])
        d1 = downsample_old(tmp_data, 8, True)
        d2 = blocks.downsample(tmp_data, 8, 3)
        assert(np.sum(np.abs(d1-d2))/np.sum(np.abs(d1))<1e-6)

    def test_downsampler_2d(self):
        # Testing up_sampler, down_sampler
        x = tf.placeholder(tf.float32, shape=[1,256,256,1],name='x')
        input_img = np.reshape(np.random.randn(256,256), [1,256,256,1])
        xd = blocks.down_sampler(x, s=2)
        xdu = blocks.up_sampler(xd, s=2)
        xdud = blocks.down_sampler(xdu, s=2)
        xdudu = blocks.up_sampler(xdud, s=2)
        with tf.Session() as sess:
            img_d, img_du = sess.run([xd,xdu], feed_dict={x: input_img})
            img_dud, img_dudu = sess.run([xdud,xdudu], feed_dict={x: input_img})
        img_d = np.squeeze(img_d)
        img_du = np.squeeze(img_du)
        img_dud = np.squeeze(img_dud)
        img_dudu = np.squeeze(img_dudu)
        img = np.squeeze(input_img)

        img_d2 = np.zeros([128,128])

        for i in range(128):
            for j in range(128):
                img_d2[i,j] = np.mean(img[2*i:2*(i+1),2*j:2*(j+1)])
        assert(np.linalg.norm(img_d2-img_d,ord='fro')<1e-4)
        assert(np.linalg.norm(img_dud-img_d,ord='fro')<1e-4)
        assert(np.linalg.norm(img_dudu-img_du,ord='fro')<1e-4)


    def test_downsampler_3d(self):
        # Testing up_sampler, down_sampler
        x = tf.placeholder(tf.float32, shape=[1,64,64, 64,1],name='x')
        input_img = np.reshape(np.random.randn(64,64, 64), [1,64,64, 64,1])
        xd = blocks.down_sampler(x, s=2, size=3)
        xdu = blocks.up_sampler(xd, s=2, size=3)
        xdud = blocks.down_sampler(xdu, s=2, size=3)
        xdudu = blocks.up_sampler(xdud, s=2, size=3)
        with tf.Session() as sess:
            img_d, img_du = sess.run([xd,xdu], feed_dict={x: input_img})
            img_dud, img_dudu = sess.run([xdud,xdudu], feed_dict={x: input_img})
        img_d = np.squeeze(img_d)
        img_du = np.squeeze(img_du)
        img_dud = np.squeeze(img_dud)
        img_dudu = np.squeeze(img_dudu)
        img = np.squeeze(input_img)

        img_d2 = np.zeros([32,32,32])

        for i in range(32):
            for j in range(32):
                for k in range(32):
                    img_d2[i,j,k] = np.mean(img[2*i:2*(i+1),2*j:2*(j+1),2*k:2*(k+1)])
        assert(np.linalg.norm((img_d2-img_d).flatten())<1e-4)
        assert(np.linalg.norm((img_dud-img_d).flatten())<1e-4)
        assert(np.linalg.norm((img_dudu-img_du).flatten())<1e-4)

    def test_downsample_np(self):
        x = np.random.rand(25, 64).astype(np.float32)
        scaling = 4
        x1 = blocks.downsample(x, scaling)
        with tf.Session() as sess:
            x2 = np.squeeze(blocks.down_sampler(np.reshape(x, [*x.shape, 1]), scaling).eval())
        np.testing.assert_allclose(x1, x2, atol=1e-6)

        x = np.random.rand(25, 64, 64).astype(np.float32)
        scaling = 4
        x1 = blocks.downsample(x, scaling)
        with tf.Session() as sess:
            x2 = np.squeeze(blocks.down_sampler(np.reshape(x, [*x.shape, 1]), scaling).eval())
        np.testing.assert_allclose(x1, x2, atol=1e-6)

        x = np.random.rand(25, 32, 32, 32).astype(np.float32)
        scaling = 4
        x1 = blocks.downsample(x, scaling)
        with tf.Session() as sess:
            x2 = np.squeeze(blocks.down_sampler(np.reshape(x, [*x.shape, 1]), scaling).eval())
        np.testing.assert_allclose(x1, x2, atol=1e-6)

if __name__ == '__main__':
    unittest.main()