import tensorflow as tf
import numpy as np
# The next import should be changed
from gantools.blocks import * 
from gantools import utils
from tfnntools.model import BaseNet, rprint
from gantools.plot import colorize
from gantools.metric import ganlist
from gantools.data.transformation import tf_flip_slices, tf_patch2img
from gantools.plot.plot_summary import PlotSummaryPlot
from copy import deepcopy

class BaseGAN(BaseNet):
    """Abstract class for the model."""
    def __init__(self, params={}, name='BaseGAN'):
        self.G_fake = None
        self.D_real = None
        self.D_fake = None
        self._D_loss = None
        self._G_loss = None
        self._summary = None
        self._constraints = []
        super().__init__(params=params, name=name)
        self._loss = (self.D_loss, self.G_loss)

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def D_loss(self):
        return self._D_loss

    @property
    def G_loss(self):
        return self._G_loss

    @property
    def summary(self):
        return self._summary

    @property
    def has_encoder(self):
        return False

    @property
    def constraints(self):
        return self._constraints
    

    def sample_latent(self, N):
        raise NotImplementedError("This is a an abstract class.")


class WGAN(BaseGAN):
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [16, 16, 1] # Shape of the image
        d_params['prior_distribution'] = 'gaussian' # prior distribution
        d_params['gamma_gp'] = 10 
        d_params['loss_type'] = 'wasserstein'  # 'hinge' or 'wasserstein'

        bn = False

        d_params['generator'] = dict()
        d_params['generator']['latent_dim'] = 100
        d_params['generator']['full'] = [2*8 * 8]
        d_params['generator']['nfilter'] = [2, 32, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        d_params['generator']['stride'] = [1, 2, 1, 1]
        d_params['generator']['summary'] = True
        d_params['generator']['data_size'] = 2 # 1 for 1D signal, 2 for images, 3 for 3D
        d_params['generator']['inception'] = False # Use inception module
        d_params['generator']['residual'] = False # Use residual connections
        d_params['generator']['activation'] = lrelu # leaky relu
        d_params['generator']['one_pixel_mapping'] = [] # One pixel mapping
        d_params['generator']['non_lin'] = tf.nn.relu # non linearity at the end of the generator
        d_params['generator']['spectral_norm'] = False # use spectral norm

        d_params['discriminator'] = dict()
        d_params['discriminator']['full'] = [32]
        d_params['discriminator']['nfilter'] = [16, 32, 32, 32]
        d_params['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        d_params['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3]]
        d_params['discriminator']['stride'] = [2, 2, 2, 1]
        d_params['discriminator']['summary'] = True
        d_params['discriminator']['data_size'] = 2 # 1 for 1D signal, 2 for images, 3 for 3D
        d_params['discriminator']['inception'] = False # Use inception module
        d_params['discriminator']['activation'] = lrelu # leaky relu
        d_params['discriminator']['one_pixel_mapping'] = [] # One pixel mapping
        d_params['discriminator']['non_lin'] = None # non linearity at the beginning of the discriminator
        d_params['discriminator']['cdf'] = None # cdf
        d_params['discriminator']['cdf_block'] = None # non linearity at the beginning of the discriminator
        d_params['discriminator']['moment'] = None # non linearity at the beginning of the discriminator
        d_params['discriminator']['minibatch_reg'] = False # Use minibatch regularization
        d_params['discriminator']['spectral_norm'] = False # use spectral norm


        return d_params

    def __init__(self, params, name='wgan'):
        super().__init__(params=params, name=name)
        self._summary = tf.summary.merge(tf.get_collection("model"))

    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        self.X_fake = self.generator(self.z, reuse=False)

    def _build_net(self):
        self._data_size = self.params['generator']['data_size']
        assert(self.params['discriminator']['data_size'] == self.data_size)
        
        reduction = stride2reduction(self.params['generator']['stride'])
        if 'in_conv_shape' not in self.params['generator'].keys():
            in_conv_shape = [el//reduction for el in self.params['shape'][:-1]]
            self._params['generator']['in_conv_shape'] = in_conv_shape
  
        self._build_generator()
        self._D_fake = self.discriminator(self.X_fake, reuse=False)
        self._D_real = self.discriminator(self.X_real, reuse=True)
        self._D_loss_f = tf.reduce_mean(self._D_fake)
        self._D_loss_r = tf.reduce_mean(self._D_real)

        if self.params['loss_type'] == 'wasserstein':
            # Wasserstein loss
            gamma_gp = self.params['gamma_gp']
            print(' Wasserstein loss with gamma_gp={}'.format(gamma_gp))
            self._D_gp = self.wgan_regularization(gamma_gp, [self.X_fake], [self.X_real])
            self._D_loss = -(self._D_loss_r - self._D_loss_f) + self._D_gp
            self._G_loss = -self._D_loss_f
        elif self.params['loss_type'] == 'normalized_wasserstein':            # Wasserstein loss
            gamma_gp = self.params['gamma_gp']
            print(' Normalized Wasserstein loss with gamma_gp={}'.format(gamma_gp))
            self._D_gp = self.wgan_regularization(gamma_gp, [self.X_fake], [self.X_real])
            reg = tf.nn.relu(self._D_loss_r*self._D_loss_f)
            self._D_loss = -(self._D_loss_r - self._D_loss_f) + self._D_gp + reg
            self._G_loss = -self._D_loss_f
            tf.summary.scalar("Disc/reg", reg, collections=["train"])
        elif self.params['loss_type'] == 'hinge':
            # Hinge loss
            print(' Hinge loss.')
            self._D_loss = tf.nn.relu(1-self._D_loss_r) + tf.nn.relu(self._D_loss_f+1)
            self._G_loss = -self._D_loss_f
        else:
            raise ValueError('Unknown loss type!')    
        self._inputs = (self.z)
        self._outputs = (self.X_fake)


    def _add_summary(self):
        tf.summary.histogram('Prior/z', self.z, collections=['model'])
        self._build_image_summary()
        self._build_stat_summary()
        self._wgan_summaries()

    def generator(self, z, **kwargs):
        return generator(z, params=self.params['generator'], **kwargs)

    def discriminator(self, X, **kwargs):
        return discriminator(X, params=self.params['discriminator'], **kwargs) 

    def sample_latent(self, bs=1):
        latent_dim = self.params['generator']['latent_dim']
        return utils.sample_latent(bs, latent_dim, self._params['prior_distribution'])

    def wgan_regularization(self, gamma, list_fake, list_real):
        if not gamma:
            # I am not sure this part or the code is still useful
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
            self._constraints.append(D_clip)
            D_gp = tf.constant(0, dtype=tf.float32)
            print(" [!] Using weight clipping")
        else:
            # calculate `x_hat`
            assert(len(list_fake) == len(list_real))
            bs = tf.shape(list_fake[0])[0]
            eps = tf.random_uniform(shape=[bs], minval=0, maxval=1)

            x_hat = []
            for fake, real in zip(list_fake, list_real):
                singledim = [1]* (len(fake.shape.as_list())-1)
                eps = tf.reshape(eps, shape=[bs,*singledim])
                x_hat.append(eps * real + (1.0 - eps) * fake)

            D_x_hat = self.discriminator(*x_hat, reuse=True)

            # gradient penalty
            gradients = tf.gradients(D_x_hat, x_hat)
            norm_gradient_pen = tf.norm(gradients[0], ord=2)
            D_gp = gamma * tf.square(norm_gradient_pen - 1.0)
            tf.summary.scalar("Disc/GradPen", D_gp, collections=["train"])
            tf.summary.scalar("Disc/NormGradientPen", norm_gradient_pen, collections=["train"])
            print(" Using gradients penalty")

        return D_gp

    def _wgan_summaries(self):
        tf.summary.scalar("Disc/Neg_Loss", -self._D_loss, collections=["train"])
        tf.summary.scalar("Disc/Neg_Critic", self._D_loss_f - self._D_loss_r, collections=["train"])
        tf.summary.scalar("Disc/Loss_f", self._D_loss_f, collections=["train"])
        tf.summary.scalar("Disc/Loss_r", self._D_loss_r, collections=["train"])
        tf.summary.scalar("Gen/Loss", self._G_loss, collections=["train"])
   
    def _build_stat_summary(self):
        self._stat_list_real = ganlist.gan_stat_list('real')
        self._stat_list_fake = ganlist.gan_stat_list('fake')

        for stat in self._stat_list_real:
            stat.add_summary(collections="model")

        for stat in self._stat_list_fake:
            stat.add_summary(collections="model")

        self._metric_list = ganlist.gan_metric_list(size=self.data_size)
        for met in self._metric_list:
            met.add_summary(collections="model")

    def preprocess_summaries(self, X_real, **kwargs):
        for met in self._metric_list:
            met.preprocess(X_real, **kwargs)

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        for stat in self._stat_list_real:
            feed_dict = stat.compute_summary(X_real, feed_dict)
        for stat in self._stat_list_fake:
            feed_dict = stat.compute_summary(X_fake, feed_dict)
        for met in self._metric_list:
            feed_dict = met.compute_summary(X_fake, X_real, feed_dict)
        if self.data_size==1:
            feed_dict = self._plot_real.compute_summary(np.squeeze(X_real), feed_dict=feed_dict)
            feed_dict = self._plot_fake.compute_summary(np.squeeze(X_fake), feed_dict=feed_dict)
        return feed_dict

    def _build_image_summary(self):
        vmin = tf.reduce_min(self.X_real)
        vmax = tf.reduce_max(self.X_real)
        if self.data_size==3:
            X_real = utils.tf_cube_slices(self.X_real)
            X_fake = utils.tf_cube_slices(self.X_fake)
            # Plot some slices
            sl = self.X_real.shape[3]//2
            tf.summary.image(
                "images/Real_Image_slice_middle",
                colorize(self.X_real[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            tf.summary.image(
                "images/Fake_Image_slice_middle",
                colorize(self.X_fake[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            sl = self.X_real.shape[3]-1
            tf.summary.image(
                "images/Real_Image_slice_end",
                colorize(self.X_real[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            tf.summary.image(
                "images/Fake_Image_slice_end",
                colorize(self.X_fake[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            sl = (self.X_real.shape[3]*3)//4
            tf.summary.image(
                "images/Real_Image_slice_3/4",
                colorize(self.X_real[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
            tf.summary.image(
                "images/Fake_Image_slice_3/4",
                colorize(self.X_fake[:,:,:,sl,:], vmin, vmax),
                max_outputs=4,
                collections=['model'])
        elif self.data_size==2:
            X_real = self.X_real
            X_fake = self.X_fake
        elif self.data_size==1:
            self._plot_real = PlotSummaryPlot(4, 4, "real", "signals", collections=['model'])
            self._plot_fake = PlotSummaryPlot(4, 4, "fake", "signals", collections=['model'])
            fs = self.params.get('fs', 16000)
            tf.summary.audio(
                'audio/Real', self.X_real, fs, max_outputs=4, collections=['model'])
            tf.summary.audio(
                'audio/Fake', self.X_fake, fs, max_outputs=4, collections=['model'])
            return None
        tf.summary.image(
            "images/Real_Image",
            colorize(X_real, vmin, vmax),
            max_outputs=4,
            collections=['model'])
        tf.summary.image(
            "images/Fake_Image",
            colorize(X_fake, vmin, vmax),
            max_outputs=4,
            collections=['model'])

    def assert_image(self, x):
        dim = self.data_size + 1
        if len(x.shape) < dim:
            raise ValueError('The size of the data is wrong')
        elif len(x.shape) < (dim +1):
            x = np.expand_dims(x, dim)
        return x

    def batch2dict(self, batch):
        d = dict()
        d['X_real'] = self.assert_image(batch)
        d['z'] = self.sample_latent(len(batch))
        return d

    @property
    def data_size(self):
        return self._data_size

class SpectrogramGAN(WGAN):
    def default_params(self):
        d_params = super().default_params()
        d_params['generator']['consistency_contribution'] = 0
        return d_params
    
    def substractMeanAndDivideByStd(self, aDistribution):
        unmeaned = aDistribution -tf.reduce_mean(aDistribution, axis=(1, 2), keep_dims=True)     
        shiftedtt = unmeaned/tf.sqrt(tf.reduce_sum(tf.square(tf.abs(unmeaned)), axis=(1, 2), keep_dims=True))
        return shiftedtt
            
    def consistency(self, spectrogram):
        ttderiv = spectrogram[:, 1:-1, :-2] - 2*spectrogram[:, 1:-1, 1:-1] + spectrogram[:, 1:-1, 2:] + tf.constant(np.pi / 4)
        ffderiv = spectrogram[:, :-2, 1:-1] - 2*spectrogram[:, 1:-1, 1:-1] + spectrogram[:, 2:, 1:-1] + tf.constant(np.pi / 4) 

        absttderiv = self.substractMeanAndDivideByStd(tf.abs(ttderiv))
        absffderiv = self.substractMeanAndDivideByStd(tf.abs(ffderiv))
        
        consistencies = tf.reduce_sum(absttderiv*absffderiv, axis=(1,2))
        return consistencies
    
    def _build_net(self):
        super()._build_net()
        consistency_contribution = self._params['generator']['consistency_contribution']
        print("consistency_contribution", consistency_contribution)
        self._R_Con = self.consistency((self.X_real-1)*5)
        self._F_Con = self.consistency((self.X_fake-1)*5)
        self._mean_R_Con, self._std_R_Con = tf.nn.moments(self._R_Con, axes=[0])
        self._mean_F_Con, self._std_F_Con = tf.nn.moments(self._F_Con, axes=[0])
        
        self._mean_R_Con = tf.squeeze(self._mean_R_Con)
        self._mean_F_Con = tf.squeeze(self._mean_F_Con)
        self._std_R_Con = tf.squeeze(self._std_R_Con)
        self._std_F_Con = tf.squeeze(self._std_F_Con)
                
        self._G_Reg =  tf.abs(self._mean_R_Con - self._mean_F_Con)
        self._G_loss += consistency_contribution*self._G_Reg

    def _wgan_summaries(self):
        super()._wgan_summaries()
        tf.summary.scalar("Gen/Reg", self._G_Reg, collections=["train"])   
        tf.summary.scalar("Gen/R_Con", self._mean_R_Con, collections=["train"])   
        tf.summary.scalar("Gen/F_Con", self._mean_F_Con, collections=["train"])   

        tf.summary.scalar("Gen/STD_diff", tf.abs(self._std_R_Con-self._std_F_Con), collections=["train"])   
        tf.summary.scalar("Gen/R_STD_Con", self._std_R_Con, collections=["train"])   
        tf.summary.scalar("Gen/F_STD_Con", self._std_F_Con, collections=["train"])  

        
class DiffSpectrogramGAN(SpectrogramGAN):   
    def secondDerivs(self, spectrogram):
        ttderiv = spectrogram[:, 1:-1, :-2] - 2*spectrogram[:, 1:-1, 1:-1] + spectrogram[:, 1:-1, 2:] + tf.constant(np.pi / 4)
        ffderiv = spectrogram[:, :-2, 1:-1] - 2*spectrogram[:, 1:-1, 1:-1] + spectrogram[:, 2:, 1:-1] + tf.constant(np.pi / 4)
        normalizedttderiv = (ttderiv-3)/10
        normalizedffderiv = (ffderiv-3)/10
        return tf.pad(normalizedttderiv, [[0, 0], [1, 1,], [1, 1], [0, 0]]), tf.pad(normalizedffderiv, [[0, 0], [1, 1,], [1, 1], [0, 0]])

    def firstDerivs(self, spectrogram):
        tderiv = (spectrogram[:, 1:-1, 2:] - spectrogram[:, 1:-1, :-2])/2
        fderiv = (spectrogram[:, 2:, 1:-1] - spectrogram[:, :-2, 1:-1])/2
        normalizedtderiv = tderiv/4
        normalizedfderiv = fderiv/4
        return tf.pad(normalizedtderiv, [[0, 0], [1, 1,], [1, 1], [0, 0]]), tf.pad(normalizedfderiv, [[0, 0], [1, 1,], [1, 1], [0, 0]])
               
    def discriminator(self, X, **kwargs):
        return discriminator(tf.concat([X, *self.firstDerivs((X-1)*5), *self.secondDerivs((X-1)*5)], axis=-1), params=self.params['discriminator'], **kwargs) 
        
class InpaintingGAN(WGAN):
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['inpainting'] = dict()
        v = 4096 + 2048
        d_params['inpainting']['split'] = [v, 4096, v]
        return d_params

    def __init__(self, params, name='inpaint_gan'):
        # Only works with 1D signal for now
        assert(params['generator']['data_size'] in [1,2])
        super().__init__(params=params, name=name)
        self._inputs = (self.z, self.borders)
        self._outputs = (self.X_fake)

    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        borderleft, self.center_real, borderright = tf.split(self.X_real, self.params['inpainting']['split'], axis=2)
        borders = tf.concat([borderleft,borderright], axis=self.data_size+1)
        inshape = borders.shape.as_list()[1:]

        self.borders = tf.placeholder_with_default(borders, shape=[None, *inshape], name='borders')


        self.X_fake_center = self.generator(self.z,  y=self.borders, reuse=False)
        # Those line should be done in a better way
        if self.data_size == 1:
            self.X_fake = tf.concat([self.borders[:,:,0:1], self.X_fake_center, self.borders[:,:,1:2]], axis=1)
        elif self.data_size == 2:
            self.X_fake = tf.concat([self.borders[:,:,:,0:1], self.X_fake_center, self.borders[:,:,:,1:2]], axis=2)
        else:
            raise NotImplementedError()

    def generator(self, z, y, **kwargs):
        return generator_border(z, y=y, params=self.params['generator'], **kwargs)


class CosmoWGAN(WGAN):
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['cosmology'] = dict()
        d_params['cosmology']['forward_map'] = None
        d_params['cosmology']['backward_map'] = None
        return d_params

    def _build_stat_summary(self):
        super()._build_stat_summary()
        self._cosmo_metric_list = ganlist.cosmo_metric_list()
        for met in self._cosmo_metric_list:
            met.add_summary(collections="model")

    def preprocess_summaries(self, X_real, **kwargs):
        super().preprocess_summaries(X_real, **kwargs)
        if self.params['cosmology']['backward_map']:
            X_real = self.params['cosmology']['backward_map'](X_real)
        for met in self._cosmo_metric_list:
            met.preprocess(X_real, **kwargs)

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        feed_dict = super().compute_summaries(X_real, X_fake, feed_dict)
        if self.params['cosmology']['backward_map']:
            if X_real is not None:
                X_real = self.params['cosmology']['backward_map'](X_real)
            X_fake = self.params['cosmology']['backward_map'](X_fake)
        for met in self._cosmo_metric_list:
            feed_dict = met.compute_summary(X_fake, X_real, feed_dict)
        return feed_dict


class LapWGAN(WGAN):
    # TODO add summaries for the 1D case...
    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [32, 32, 1] # Shape of the image
        bn = False
        d_params['upscaling'] = 2
        d_params['generator']['latent_dim'] = 32*32
        d_params['generator']['full'] = []
        d_params['generator']['nfilter'] = [16, 32, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        d_params['generator']['stride'] = [1, 1, 1, 1]

        return d_params

    def __init__(self, params, name='lap_wgan'):
        super().__init__(params=params, name=name)

    def _build_generator(self):
        shape = self._params['shape']
        self.X_real = tf.placeholder(tf.float32, shape=[None, *shape], name='Xreal')
        self.upscaling = self.params['upscaling']
        X_down = down_sampler(self.X_real, s=self.upscaling)
        inshape = X_down.shape.as_list()[1:]
        self.X_down = tf.placeholder_with_default(X_down, shape=[None, *inshape], name='y')
        self.X_smooth = up_sampler(self.X_down, s=self.upscaling, smoothout=True)
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        self.X_fake = self.generator(self.z, X=self.X_smooth, reuse=False)

    def discriminator(self, X, **kwargs):
        axis = self.data_size + 1
        v = tf.concat([X, self.X_smooth, X-self.X_smooth], axis=axis)
        return discriminator(v, params=self.params['discriminator'], **kwargs)

    def _build_image_summary(self):
        super()._build_image_summary()
        vmin = tf.reduce_min(self.X_real)
        vmax = tf.reduce_max(self.X_real)
        if self.data_size==3:
            X_smooth = utils.tf_cube_slices(self.X_smooth)
        elif self.data_size==2:
            X_smooth = self.X_smooth

        elif self.data_size==1:
            self._plot_down = PlotSummaryPlot(4, 4, "down", "signals", collections=['model'])
            fs = self.params.get('fs', 16000)
            tf.summary.audio(
                'audio/down', self.X_smooth, fs, max_outputs=4, collections=['model'])
            return None
        tf.summary.image(
            "images/Down_sampled_image",
            colorize(X_smooth, vmin, vmax),
            max_outputs=4,
            collections=['model'])

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        feed_dict = super().compute_summaries(X_real, X_fake, feed_dict)
        if self.data_size==1:
            X_smooth = self.X_smooth.eval(feed_dict=feed_dict)
            feed_dict = self._plot_down.compute_summary(np.squeeze(X_smooth), feed_dict=feed_dict)
        return feed_dict

class UpscalePatchWGAN(WGAN):
    '''
    Generate blocks, using top, left and top-left border information
    '''

    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [16, 16, 4] # Shape of the input data (1 image and 3 borders)
        bn = False
        d_params['upscaling'] = None
        d_params['generator']['latent_dim'] = 16*16
        d_params['generator']['full'] = []
        d_params['generator']['nfilter'] = [16, 32, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        d_params['generator']['stride'] = [1, 1, 1, 1]
        d_params['generator']['use_old_gen'] = False

        return d_params

    def __init__(self, params, name='upscale_patch_wgan'):
        super().__init__(params=params, name=name)
        assert(not(self.params['upscaling']==1))
        if self.params['upscaling']:
            self._inputs = (self.z, self.borders)
        self._outputs = (self.X_fake_corner)


    def _build_generator(self):
        shape = self.params['shape']
        self.X_data = tf.placeholder(tf.float32, shape=[None, *shape], name='X_data')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        # A) Separate real data and border information
        if self.data_size==3:
            axis = 4
            o = 7
        elif self.data_size==2:
            axis = 3
            o = 3
        else:
            axis=2
            o = 1
        la = shape[-1]
        mult = la//(1+o)
        self.X_real_corner, borders = tf.split(self.X_data, [mult,o*mult], axis=axis)
        inshape = borders.shape.as_list()[1:]
        self.borders = tf.placeholder_with_default(borders, shape=[None, *inshape], name='borders')
        
        # B) Split the borders
        border_list = tf.split(self.borders, o, axis=axis)
        # D) Flip the borders
        flipped_border_list = tf_flip_slices(*border_list, size=self.data_size)

        # C) Handling downsampling
        if self.params['upscaling']:
            self.upscaling = self.params['upscaling']
            X_down = down_sampler(self.X_real_corner, s=self.upscaling)
            inshape = X_down.shape.as_list()[1:]
            self.X_down = tf.placeholder_with_default(X_down, shape=[None, *inshape], name='y')
            self.X_smooth = up_sampler(self.X_down, s=self.upscaling)
            self.X_fake_corner = self.generator(z=self.z, y=flipped_border_list, X=self.X_smooth, reuse=False)

        else:
            self.X_smooth = None
            # E) Generater the corner
            self.X_fake_corner = self.generator(z=self.z, y=flipped_border_list, reuse=False)



        
        #F) Recreate the big images
        self.X_real = tf_patch2img(self.X_real_corner, *border_list, size=self.data_size)
        self.X_fake = tf_patch2img(self.X_fake_corner, *border_list, size=self.data_size)

        if self.params['upscaling']:
            self.X_down_up = up_sampler(down_sampler(self.X_real, s=self.upscaling),s=self.upscaling)

    def discriminator(self, X, **kwargs):
        if self.params['upscaling']:
            axis = self.data_size + 1
            v = tf.concat([X, self.X_down_up, X-self.X_down_up], axis=axis)
        else:
            v = X
        return discriminator(v, params=self.params['discriminator'], **kwargs)

    def _get_corner(self, X):
        if X is not None:
            axis = self.data_size + 1
            slc = [slice(None)] * len(X.shape)
            slc[axis] = 0
            return X[slc]

    def preprocess_summaries(self, X_real, **kwargs):
        super().preprocess_summaries(self._get_corner(self.assert_image(X_real)), **kwargs)

    def compute_summaries(self, X_real, X_fake, feed_dict={}):
        feed_dict = super().compute_summaries(self._get_corner(X_real), self._get_corner(X_fake), feed_dict)
        if self.data_size==1 and self.params['upscaling']:
            X_smooth = self.X_smooth.eval(feed_dict=feed_dict)
            feed_dict = self._plot_down.compute_summary(np.squeeze(X_smooth), feed_dict=feed_dict)
        return feed_dict

    def batch2dict(self, batch):
        d = dict()
        d['X_data'] = self.assert_image(batch)
        d['z'] = self.sample_latent(len(batch))
        return d


    def _build_image_summary(self):
        super()._build_image_summary()
        if self.params['upscaling']:
            vmin = tf.reduce_min(self.X_real)
            vmax = tf.reduce_max(self.X_real)
            if self.data_size==3:
                X_smooth = utils.tf_cube_slices(self.X_smooth)
            elif self.data_size==2:
                X_smooth = self.X_smooth
            elif self.data_size==1:
                self._plot_down = PlotSummaryPlot(4, 4, "down", "signals", collections=['model'])
                fs = self.params.get('fs', 16000)
                tf.summary.audio(
                    'audio/down', self.X_smooth, fs, max_outputs=4, collections=['model'])
                return None
            tf.summary.image(
                "images/Down_sampled_image",
                colorize(X_smooth, vmin, vmax),
                max_outputs=4,
                collections=['model'])

    def generator(self, z, y, X=None, **kwargs):
#         if self.params['generator']['use_old_gen']:
#             print('Using old generator...')
#             return generator_up(z, X=X, y=y, params=self.params['generator'], **kwargs, scope='generator')
#         else:
        if self.params['generator'].get('borders', None):
            axis = self.data_size + 1
            newy = tf.concat(y, axis=axis)
            return generator_border(z, X=X, y=newy, params=self.params['generator'], **kwargs)
        else:
            axis = self.data_size +1
            if self.params['upscaling']:
                if self.data_size==1:
                    # y = remove_center(y, self.data_size)
                    newX = tf.concat([X, y], axis=axis)
                else:
                    newX = tf.concat([X, *y], axis=axis)
            else:
                newX = tf.concat(y, axis=axis)
            return generator(z, X=newX, params=self.params['generator'], **kwargs)

# def remove_center(X, data_size):
#     '''Only keep the last pixel and set the center to 0.'''
#     zt = np.zeros([1,*X.shape[1:data_size+1], 1], dtype=tf.float32)
#     for i in range(data_size):
#         axis = i + 1
#         slc = [slice(None)] * len(X.shape)
#         slc[axis] = slice(0,X.shape[1+i], X.shape[1+i]-1)
#         zt[slc] = 1
#     zt = tf.convert_to_tensor(zt, np.float32)
#     return X*zt

# class UpscalePatchWGANBordersOld(UpscalePatchWGAN):
#     '''
#     Generate blocks, using top, left and top-left border information

#     This model will encode borders instead of flipping them.
#     '''

#     def default_params(self):
#         d_params = deepcopy(super().default_params())
#         bn = False
#         d_params['generator']['full'] = [512]
#         d_params['generator']['nfilter'] = [2, 32, 32, 1]
#         d_params['generator']['borders'] = dict()
#         d_params['generator']['borders']['width_full'] = None
#         d_params['generator']['borders']['nfilter'] = [2, 8, 1]
#         d_params['generator']['borders']['batch_norm'] = [bn, bn, bn]
#         d_params['generator']['borders']['shape'] = [[5, 5], [5, 5], [5, 5]]
#         d_params['generator']['borders']['stride'] = [2, 2, 2]
#         d_params['generator']['borders']['data_size'] = 2 # 1 for 1D signal, 2 for images, 3 for 3D
#         return d_params


#     def generator(self, z, y=None, **kwargs):
#         axis = self.data_size +1
#         newy = tf.concat(y, axis=axis)
#         return generator_border(z, y=newy, params=self.params['generator'], **kwargs)


class UpscalePatchWGANBorders(UpscalePatchWGAN):
    '''
    Generate blocks, using top, left and top-left border information

    This model will encode borders instead of flipping them.
    '''

    def default_params(self):
        d_params = super().default_params()
        bn = False
        d_params['shape'] = [256, 1] # Shape of the image

        d_params['generator']['latent_dim'] = 16
        d_params['generator']['full'] = [32]
        d_params['generator']['nfilter'] = [2, 32, 1]
        d_params['generator']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['shape'] = [[21], [21], [21], [21]]
        d_params['generator']['stride'] = [1, 2, 2, 1]
        d_params['generator']['full'] = [64]
        d_params['generator']['nfilter'] = [2, 32, 32, 1]
        d_params['generator']['data_size'] = 1 # 1 for 1D signal, 2 for images, 3 for 3D

        d_params['generator']['borders'] = dict()
        d_params['generator']['borders']['width_full'] = None
        d_params['generator']['borders']['nfilter'] = [2, 8, 1]
        d_params['generator']['borders']['batch_norm'] = [bn, bn, bn]
        d_params['generator']['borders']['shape'] = [[5, 5], [5, 5], [5, 5]]
        d_params['generator']['borders']['stride'] = [2, 2, 2]
        d_params['generator']['borders']['data_size'] = 1 # 1 for 1D signal, 2 for images, 3 for 3D

        d_params['discriminator']['full'] = [32]
        d_params['discriminator']['nfilter'] = [16, 32, 32, 32]
        d_params['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        d_params['discriminator']['shape'] = [[21], [21], [21], [21]]
        d_params['discriminator']['stride'] = [2, 2, 2, 1]
        d_params['discriminator']['data_size'] = 1 # 1 for 1D signal, 2 for images, 3 for 3D



        return d_params

    def _build_generator(self):
        shape = self.params['shape']
        reduction = np.prod(np.array(self.params['generator']['stride']))*2
        in_conv_shape = [el//reduction for el in shape[:-1]]
        self._params['generator']['in_conv_shape'] = in_conv_shape
        self.X_data = tf.placeholder(tf.float32, shape=[None, *shape], name='X_data')
        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.params['generator']['latent_dim']],
            name='z')
        # A) Separate real data and border information
        if self.data_size==3:
            axis = 4
            o = 7
        elif self.data_size==2:
            axis = 3
            o = 3
        else:
            axis=2
            o = 1
        if self.params['upscaling']:
            self.X_real, self.X_down_up = tf.split(self.X_data, [1,1], axis=axis)
            if self.data_size==1:
                middle = self.X_down_up.shape.as_list()[1]//2
                X_smooth = self.X_down_up[:,middle:,:]
            else:
                raise NotImplementedError()
            inshape = X_smooth.shape.as_list()[1:]
            self.X_smooth = tf.placeholder_with_default(X_smooth, shape=[None, *inshape], name='y')
        else:
            self.X_real = self.X_data
            self.X_smooth = None

        if self.data_size==1:
            middle = self.X_real.shape.as_list()[1]//2
            self.X_real_corner = self.X_real[:,middle:,:]
            borders = self.X_real[:,:middle,:]
        else:
            raise NotImplementedError()
        inshape = borders.shape.as_list()[1:]
        self.borders = tf.placeholder_with_default(borders, shape=[None, *inshape], name='borders')
        
        # B) Split the borders
        border_list = tf.split(self.borders, o, axis=axis)

        # D) Flip the borders

        flipped_border_list = tf_flip_slices(*border_list, size=self.data_size)

        # E) Generater the corner
        self.X_fake_corner = self.generator(z=self.z, y=flipped_border_list, X=self.X_smooth, reuse=False)
        
        #F) Recreate the big images
        self.X_real = tf_patch2img(self.X_real_corner, *border_list, size=self.data_size)
        self.X_fake = tf_patch2img(self.X_fake_corner, *border_list, size=self.data_size)


    def generator(self, z, y=None, **kwargs):
        axis = self.data_size +1
        newy = tf.concat(y, axis=axis)
        return generator_border(z, y=newy, params=self.params['generator'], **kwargs)


# class GanModel(object):
#     ''' Abstract class for the model'''
#     def __init__(self, params, name='gan', is_3d=False):
#         self.name = name
#         self.params = params
#         self.params['generator']['is_3d'] = is_3d
#         self.params['discriminator']['is_3d'] = is_3d    
#         self._is_3d = is_3d
#         self.G_fake = None
#         self.D_real = None
#         self.D_fake = None
#         self._D_loss = None
#         self._G_loss = None

#     def add_model_specific_inputs(self, feed_dict, phase="generate", step=0):
#         return feed_dict

#     @property
#     def D_loss(self):
#         return self._D_loss

#     @property
#     def G_loss(self):
#         return self._G_loss

#     @property
#     def is_3d(self):
#         return self._is_3d

#     @property
#     def has_encoder(self):
#         return False




# # This is for testing the (expected non positive) effect of normalization on the latent variable
# # Use of a regular WGAN if you need a simple Wasserstein GAN
# class WNGanModel(GanModel):
#     def __init__(self, params, X, z, name='wngan', is_3d=False):
#         super().__init__(params=params, name=name, is_3d=is_3d)
#         zn = tf.nn.l2_normalize(z, 1)
#         self.G_fake = self.generator(zn, reuse=False)
#         self.D_real = self.discriminator(X, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, reuse=True)
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
#         self._D_loss = D_loss_r - D_loss_f + D_gp
#         self._G_loss = D_loss_f
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

#     def generator(self, z, reuse):
#         return generator(z, self.params['generator'], reuse=reuse)

#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)


# class CondWGanModel(GanModel):
#     def __init__(self, params, X, z, name='CondWGan', is_3d=False):
#         super().__init__(params=params, name=name, is_3d=is_3d)
#         self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
#         self.G_fake = self.generator(z, reuse=False)
#         self.D_real = self.discriminator(X, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, reuse=True)
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
#         # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
#         # Min(D_loss_r - D_loss_f) = Min -D_loss_f
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

#     def generator(self, z, reuse):
#         return generator(z, self.params['generator'], y=self.y, reuse=reuse)

#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], z=self.y, reuse=reuse)


# # Legacy model class required by certain old models
# class TemporalGanModelv3(GanModel):
#     def __init__(self, params, X, z, name='TempWGanV3', is_3d=False):
#         super().__init__(params=params, name=name, is_3d=is_3d)
#         assert 'time' in params.keys()

#         zn = tf.nn.l2_normalize(z, 1) * np.sqrt(params['generator']['latent_dim'])
#         z_shape = tf.shape(zn)
#         scaling = np.asarray(params['time']['class_weights'])
#         gen_bs = params['optimization']['batch_size'] * params['time']['num_classes']
#         scaling = np.resize(scaling, (gen_bs, 1))
#         default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
#         self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
#         t = self.y[:z_shape[0]]
#         zn = tf.multiply(zn, t)

#         self.G_c_fake = self.generator(zn, reuse=False)
#         self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

#         if params['time']['use_diff_stats']:
#             self.disc = self.df_discriminator
#         else:
#             self.disc = self.discriminator

#         self.D_real = self.disc(X, reuse=False)
#         self.D_fake = self.disc(self.G_fake, reuse=True)

#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)

#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.disc, [self.G_fake], [X])
#         # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
#         # Min(D_loss_r - D_loss_f) = Min -D_loss_f
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

#     def reshape_time_to_channels(self, X):
#         nc = self.params['time']['num_classes']
#         lst = []
#         for i in range(nc):
#             lst.append(X[i::nc])
#         return tf.concat(lst, axis=3)

#     def generator(self, z, reuse):
#         return generator(z, self.params['generator'], reuse=reuse)

#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)

#     def df_discriminator(self, X, reuse):
#         y = X[:, :, :, 1:] - X[:, :, :, :-1]
#         return discriminator(tf.concat([X,y], axis=3), self.params['discriminator'], reuse=reuse)

#     @property
#     def D_loss(self):
#         return self._D_loss

#     @property
#     def G_loss(self):
#         return self._G_loss


# # Generic Continuous Conditional GAN class.
# # Depending on parameters difference formats for the continuous conditional GAN
# # may be changed, such as the encoding format for the latent variables.
# class TemporalGenericGanModel(GanModel):
#     def __init__(self, params, X, z, name='TempGenericGAN', is_3d=False):
#         super().__init__(params=params, name=name, is_3d=is_3d)
#         assert 'time' in params.keys()
#         assert 'model' in params['time'].keys()

#         z = self.prep_latent(z, params['time']['model'])

#         self.G_c_fake = self.generator(z, reuse=False)
#         self.G_fake = self.reshape_time_to_channels(self.G_c_fake)

#         if False: #'encoder' in params.keys():
#             channel_images = self.reshape_channels_to_images(X)
#             channel_images = self.generator(self.encoder(channel_images, reuse=False),reuse=True)
#             self.reconstructed = self.reshape_time_to_channels(channel_images)
#             self._E_loss = tf.losses.mean_squared_error(X, self.reconstructed)

#         if 'encoder' in params.keys():
#             z_recon = self.encoder(self.G_c_fake, reuse=False)
#             self._E_loss = tf.losses.mean_squared_error(z, z_recon)
#             channel_images = self.reshape_channels_to_images(X)
#             channel_images = self.generator(self.encoder(channel_images, reuse=True),reuse=True)
#             self.reconstructed = self.reshape_time_to_channels(channel_images)
#             tf.summary.scalar("Enc/Loss", self._E_loss, collections=["Training"])
#             reconstruction_loss = tf.losses.mean_squared_error(X, self.reconstructed)
#             tf.summary.scalar("Enc/Reconstruction_Loss", reconstruction_loss, collections=["Training"])
#         else:
#             self._E_loss = None

#         if params['time']['use_diff_stats']:
#             self.disc = self.df_discriminator
#         else:
#             self.disc = self.discriminator

#         self.D_real = self.disc(X, reuse=False)
#         self.D_fake = self.disc(self.G_fake, reuse=True)

#         gamma_gp = self.params['optimization']['gamma_gp']
#         if self.params['optimization'].get('JS-regularization', False):
#             print("Using JS-Regularization (Roth et al. 2017)")
#             D_gp = js_regularization(self.D_real, X, self.D_fake, self.G_fake,
#                                      params['optimization']['batch_size'])
#             global_step = tf.train.get_global_step()
#             alpha = self.params['optimization'].get('alpha', None)
#             if alpha:
#                 print("Using annealing")
#                 T = self.params['optimization']['max_T']
#                 alpha = tf.constant(alpha, dtype=tf.float32)
#                 alpha = tf.pow(alpha, tf.cast(tf.minimum(global_step, T) / T, tf.float32))
#                 D_gp = D_gp * gamma_gp * alpha
#             else:
#                 print("Not using annealing")
#                 D_gp = gamma_gp * D_gp

#             s_D_real = tf.nn.sigmoid(self.D_real)
#             s_D_fake = tf.nn.sigmoid(self.D_fake)

#             self._D_loss = tf.reduce_mean(
#                 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(s_D_real))
#                 + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(s_D_fake))) \
#                 + D_gp

#             self._G_loss = tf.reduce_mean(
#                 tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(s_D_fake)))

#             js_gan_summaries(s_D_fake, s_D_real)
#         else:
#             D_gp = 0
#             if gamma_gp != 0:
#                 D_gp = wgan_regularization(gamma_gp, self.disc, [self.G_fake], [X])

#             if params['time']['model']['relative']:
#                 D_loss_f = tf.reduce_mean(self.D_fake - self.D_real)
#                 D_loss_r = tf.reduce_mean(self.D_real - self.D_fake)

#                 self._D_loss = -D_loss_r + D_gp
#                 self._G_loss = -D_loss_f
#                 wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#             else:
#                 D_loss_f = tf.reduce_mean(self.D_fake)
#                 D_loss_r = tf.reduce_mean(self.D_real)

#                 self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#                 self._G_loss = -D_loss_f
#                 wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)

#     def reshape_time_to_channels(self, X):
#         nc = self.params['time']['num_classes']
#         lst = []
#         for i in range(nc):
#             lst.append(X[i::nc])
#         return tf.concat(lst, axis=3)

#     def reshape_channels_to_images(self, x):
#         x = tf.transpose(x, [0, 3, 1, 2])
#         x = tf.reshape(x, [tf.shape(x)[0] * x.shape[1], x.shape[2], x.shape[3]])
#         return tf.expand_dims(x, axis=-1)

#     def prep_latent(self, z, params):
#         if params['time_encoding'] == 'channel_encoding':
#             ld_width = self.params['image_size'][0]
#             for stride in self.params['generator']['stride']:
#                 ld_width = ld_width // stride

#             assert ld_width*ld_width < self.params['generator']['latent_dim']

#             z_shape = tf.shape(z)
#             z = tf.reshape(z, [z_shape[0], ld_width, ld_width, z_shape[1] // (ld_width*ld_width)])
#             scaling = np.asarray(self.params['time']['class_weights'])
#             gen_bs = self.params['optimization']['batch_size'] * self.params['time']['num_classes']
#             scaling = np.resize(scaling, (gen_bs, 1))
#             default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
#             self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
#             t = self.y[:z_shape[0]]
#             t = tf.expand_dims(t, axis=-1)
#             t = tf.expand_dims(tf.tile(t, [1, ld_width, ld_width]), axis=-1)
#             z = z[:, :, :, :-1]
#             z = tf.concat([z, t], axis=3)
#             return tf.reshape(z, z_shape)

#         if params['time_encoding'] == 'scale_full':
#             zn = tf.nn.l2_normalize(z, 1) * np.sqrt(self.params['generator']['latent_dim'])
#             z_shape = tf.shape(zn)
#             scaling = np.asarray(self.params['time']['class_weights'])
#             gen_bs = self.params['optimization']['batch_size'] * self.params['time']['num_classes']
#             scaling = np.resize(scaling, (gen_bs, 1))
#             default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
#             self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
#             t = self.y[:z_shape[0]]
#             return tf.multiply(zn, t)

#         if params['time_encoding'] == 'scale_half':
#             z0 = z[:, 0::2]
#             z1 = z[:, 1::2]
#             zn = tf.nn.l2_normalize(z1, 1) * np.sqrt(self.params['generator']['latent_dim'] / 2)
#             z_shape = tf.shape(zn)
#             scaling = np.asarray(self.params['time']['class_weights'])
#             gen_bs = self.params['optimization']['batch_size'] * self.params['time']['num_classes']
#             scaling = np.resize(scaling, (gen_bs, 1))
#             default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
#             self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
#             t = self.y[:z_shape[0]]
#             zn = tf.multiply(zn, t)
#             zn = tf.expand_dims(zn, -1)
#             z0 = tf.expand_dims(z0, -1)
#             return tf.concat([z0, zn], axis=2)

#         if params['time_encoding'] == 'late_time_encoding':
#             scaling = np.asarray(self.params['time']['class_weights'])
#             gen_bs = self.params['optimization']['batch_size'] * self.params['time']['num_classes']
#             scaling = np.resize(scaling, (gen_bs, 1))
#             default_t = tf.constant(scaling, dtype=tf.float32, name='default_t')
#             self.y = tf.placeholder_with_default(default_t, shape=[None, 1], name='t')
#             z_shape = tf.shape(z)
#             self.t = self.y[:z_shape[0]]
#             return z
#         raise ValueError(' [!] time encoding not defined')

#     def generator(self, z, reuse):
#         return generator(z, self.params['generator'], reuse=reuse)

#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)

#     def encoder(self, X, reuse):
#         return encoder(X, self.params['encoder'], latent_dim=self.params['generator']['latent_dim'], reuse=reuse)

#     def df_discriminator(self, X, reuse):
#         y = X[:, :, :, 1:] - X[:, :, :, :-1]
#         return discriminator(tf.concat([X,y], axis=3), self.params['discriminator'], reuse=reuse)

#     @property
#     def D_loss(self):
#         return self._D_loss

#     @property
#     def G_loss(self):
#         return self._G_loss

#     @property
#     def E_loss(self):
#         return self._E_loss

#     @property
#     def has_encoder(self):
#         return 'encoder' in self.params.keys()

# class WVeeGanModel(GanModel):
#     def __init__(self, params, X, z, name='veegan', is_3d=False):
#         super().__init__(params=params, name=name, is_3d=is_3d)
#         self.latent_dim = params['generator']['latent_dim']
#         self.G_fake = self.generator(z, reuse=False)
#         self.z_real = self.encoder(X=X, reuse=False)
#         self.D_real = self.discriminator(X=X, z=self.z_real, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, z=z, reuse=True)
#         self.z_fake = self.encoder(X=self.G_fake, reuse=True)
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.z_fake], [X, self.z_real])

#         e = (z - self.z_fake)
#         weight_l2 = self.params['optimization']['weight_l2']
#         reg_l2 = self.latent_dim * weight_l2
#         L2_loss = reg_l2 * tf.reduce_mean(tf.square(e))

#         self._D_loss = D_loss_f - D_loss_r + D_gp
#         self._G_loss = -D_loss_f + L2_loss
#         self._E_loss = L2_loss

#         tf.summary.scalar("Enc/Loss_l2", self._E_loss, collections=["Training"])
#         tf.summary.scalar("Gen/Loss_f", -D_loss_f, collections=["Training"])

#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, -D_loss_r)

#     def generator(self, z, reuse):
#         return generator(z, self.params['generator'], reuse=reuse)

#     def discriminator(self, X, z, reuse):
#         return discriminator(X, self.params['discriminator'], z=z, reuse=reuse)

#     def encoder(self, X, reuse):
#         return encoder(X, self.params['encoder'], self.latent_dim, reuse=reuse)

#     @property
#     def E_loss(self):
#         return self._E_loss

#     @property
#     def has_encoder(self):
#         return True




# class LapPatchWGANModel(GanModel):
#     """4 different generators, probably not a good idea. Need too much training time. Not so good results."""
#     def __init__(self, params, X, z, name='lapgan', is_3d=False):
#         ''' z must have the same dimension as X'''
#         super().__init__(params=params, name=name, is_3d=is_3d)
        
#         # A) Down sampling the image
#         self.upsampling = params['generator']['upsampling']
#         self.Xs = down_sampler(X, s=self.upsampling)

#         # The input is the downsampled image
#         inshape = self.Xs.shape.as_list()[1:]
#         self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')

#         # B) Split the image in 4 parts
#         top, bottom = tf.split(self.y, 2, axis=1)
#         self.Xs1, self.Xs2 = tf.split(top, 2, axis=2)
#         self.Xs3, self.Xs4 = tf.split(bottom, 2, axis=2)

#         # B') Split latent in 4 parts
#         # This may/should be done differently?
#         bs = tf.shape(self.y)[0]  # Batch size
#         z = tf.reshape(z, [bs, *inshape])
#         topz, bottomz = tf.split(z, 2, axis=1)
#         z1, z2 = tf.split(topz, 2, axis=2)
#         z3, z4 = tf.split(bottomz, 2, axis=2)

#         # C) Define the 4 Generators

#         self.G_fake1 = self.generator(X=self.Xs1, z=z1, reuse=False, scope='generator1')
#         y1 = tf.reverse(self.G_fake1, axis=[2])
#         self.G_fake2 = self.generator(X=self.Xs2, z=z2, y=y1, reuse=False, scope='generator2')
#         y21 = tf.reverse(self.G_fake1, axis=[1])
#         y22 = tf.reverse(self.G_fake2, axis=[1,2])
#         y2 = tf.concat([y21, y22], axis=3)
#         self.G_fake3 = self.generator(X=self.Xs3, z=z3,y=y2, reuse=False, scope='generator3')
#         y31 = tf.reverse(self.G_fake1, axis=[1,2])
#         y32 = tf.reverse(self.G_fake2, axis=[1])
#         y33 = tf.reverse(self.G_fake3, axis=[2])
#         y3 = tf.concat([y31, y32, y33], axis=3)
#         self.G_fake4 = self.generator(X=self.Xs4, z=z4, y=y3, reuse=False, scope='generator4')

#         # D) Concatenate back
#         top = tf.concat([self.G_fake1,self.G_fake2], axis=2)
#         bottom = tf.concat([self.G_fake3,self.G_fake4], axis=2)
#         self.G_fake = tf.concat([top,bottom], axis=1)

#         # E) Discriminator
#         self.Xsu = up_sampler(self.y, s=self.upsampling)
#         self.D_real = self.discriminator(X, self.Xsu, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, self.Xsu, reuse=True)

#         # F) Losses
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
#         #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
#         # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
#         # Min(D_loss_r - D_loss_f) = Min -D_loss_f
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f

#         # G) Summaries
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#         tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])
#         if True:
#             tf.summary.image("SmallerImg/G_fake1", self.G_fake1, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake2", self.G_fake2, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake3", self.G_fake3, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake4", self.G_fake4, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y1", y1, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y21", y21, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y22", y22, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y31", y31, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y32", y32, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y33", y33, max_outputs=1, collections=['Images'])

#     def generator(self, X, z, reuse, scope, y=None):
#         return generator_up(X, z, self.params['generator'], y=y, reuse=reuse, scope=scope)

#     def discriminator(self, X, Xsu, reuse):
#         return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


# class LapPatchWGANsingleModel(GanModel):
#     """Seems to work fine but is recursive, so might be a bit slow."""
#     def __init__(self, params, X, z, name='lappatchsingle', is_3d=False):
#         ''' z must have the same dimension as X'''
#         super().__init__(params=params, name=name, is_3d=is_3d)
        
#         # A) Down sampling the image
#         self.upsampling = params['generator']['upsampling']
#         self.Xs = down_sampler(X, s=self.upsampling)

#         # The input is the downsampled image
#         inshape = self.Xs.shape.as_list()[1:]
#         self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')
#         self.Xsu = up_sampler(self.y, s=self.upsampling)

#         # B) Split the image in 4 parts
#         top, bottom = tf.split(self.Xsu, 2, axis=1)
#         self.Xs1, self.Xs2 = tf.split(top, 2, axis=2)
#         self.Xs3, self.Xs4 = tf.split(bottom, 2, axis=2)

#         # B') Split latent in 4 parts
#         # This may/should be done differently?
#         bs = tf.shape(self.y)[0]  # Batch size
#         zshape = X.shape.as_list()[1:]
#         z = tf.reshape(z, [bs, *zshape])
#         topz, bottomz = tf.split(z, 2, axis=1)
#         z1, z2 = tf.split(topz, 2, axis=2)
#         z3, z4 = tf.split(bottomz, 2, axis=2)

#         # C) Define the Generator

#         tinshape = tf.shape(z1)
#         y00 = tf.fill(tinshape, -1.)
#         y0 = tf.concat([y00, y00, y00], axis=3)

#         self.G_fake1 = self.generator(X=self.Xs1, z=z1, border=y0, reuse=False, scope='generator')
#         y11 = tf.reverse(self.G_fake1, axis=[2])
#         y1 = tf.concat([y11, y00, y00], axis=3)

#         self.G_fake2 = self.generator(X=self.Xs2, z=z2, border=y1, reuse=True, scope='generator')
#         y21 = tf.reverse(self.G_fake1, axis=[1])
#         y22 = tf.reverse(self.G_fake2, axis=[1, 2])
#         y2 = tf.concat([y21, y22, y00], axis=3)

#         self.G_fake3 = self.generator(X=self.Xs3, z=z3, border=y2, reuse=True, scope='generator')
#         y31 = tf.reverse(self.G_fake1, axis=[1, 2])
#         y32 = tf.reverse(self.G_fake2, axis=[1])
#         y33 = tf.reverse(self.G_fake3, axis=[2])
#         y3 = tf.concat([y31, y32, y33], axis=3)
#         self.G_fake4 = self.generator(X=self.Xs4, z=z4, border=y3, reuse=True, scope='generator')

#         # D) Concatenate back
#         top = tf.concat([self.G_fake1, self.G_fake2], axis=2)
#         bottom = tf.concat([self.G_fake3, self.G_fake4], axis=2)
#         self.G_fake = tf.concat([top, bottom], axis=1)

#         # E) Discriminator
#         self.D_real = self.discriminator(X, self.Xsu, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, self.Xsu, reuse=True)

#         # F) Losses
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu], [X, self.Xsu])
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f

#         # G) Summaries
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#         tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Real_Diff", X - self.Xsu, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])
#         if True:
#             tf.summary.image("SmallerImg/G_fake1", self.G_fake1, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake2", self.G_fake2, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake3", self.G_fake3, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake4", self.G_fake4, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y1", y11, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y21", y21, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y22", y22, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y31", y31, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y32", y32, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y33", y33, max_outputs=1, collections=['Images'])

#     def generator(self, X, z, border, reuse, scope):
#         return generator_up(tf.concat([X, border], axis=3), z, self.params['generator'], reuse=reuse, scope=scope)

#     def discriminator(self, X, Xsu, reuse):
#         return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


# class PatchWGANsingleModel(GanModel):
#     '''
#     Divide image into 4 parts, and iterative generate them
#     '''
#     def __init__(self, params, X, z, name='patchsingle', is_3d=False):
#         ''' z must have the same dimension as X'''
#         super().__init__(params=params, name=name, is_3d=is_3d)

#         # A) Split latent in 4 parts
#         bs = tf.shape(X)[0]  # Batch size
#         # nb pixel
#         inshape = X.shape.as_list()[1:]
#         z = tf.reshape(z, [bs, *inshape])
#         topz, bottomz = tf.split(z, 2, axis=1)
#         z1, z2 = tf.split(topz, 2, axis=2)
#         z3, z4 = tf.split(bottomz, 2, axis=2)

#         tinshape = tf.shape(z1)

#         y00 = tf.fill(tinshape, -1.)
#         y0 = tf.concat([y00, y00, y00], axis=3)
#         self.G_fake1 = self.generator(z=z1, border=y0, reuse=False, scope='generator')

#         y11 = tf.reverse(self.G_fake1, axis=[2])
#         y1 = tf.concat([y11, y00, y00], axis=3)
#         self.G_fake2 = self.generator(z=z2, border=y1, reuse=True, scope='generator')

#         y21 = tf.reverse(self.G_fake1, axis=[1])
#         y22 = tf.reverse(self.G_fake2, axis=[1,2])
#         y2 = tf.concat([y21, y22, y00], axis=3)
#         self.G_fake3 = self.generator(z=z3,border=y2, reuse=True, scope='generator')

#         y31 = tf.reverse(self.G_fake1, axis=[1,2])
#         y32 = tf.reverse(self.G_fake2, axis=[1])
#         y33 = tf.reverse(self.G_fake3, axis=[2])
#         y3 = tf.concat([y31, y32, y33], axis=3)
#         self.G_fake4 = self.generator(z=z4, border=y3, reuse=True, scope='generator')

#         # B) Concatenate back
#         top = tf.concat([self.G_fake1,self.G_fake2], axis=2)
#         bottom = tf.concat([self.G_fake3,self.G_fake4], axis=2)
#         self.G_fake = tf.concat([top,bottom], axis=1)

#         # C) Discriminator
#         self.D_real = self.discriminator(X, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, reuse=True)

#         # D) Losses
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake], [X])
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f

#         # E) Summaries
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#         tf.summary.image("training/Real", X, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/G_fake", self.G_fake, max_outputs=2, collections=['Images'])
#         if True:
#             tf.summary.image("SmallerImg/G_fake1", self.G_fake1, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake2", self.G_fake2, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake3", self.G_fake3, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/G_fake4", self.G_fake4, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y1", y11, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y21", y21, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y22", y22, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y31", y31, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y32", y32, max_outputs=1, collections=['Images'])
#             tf.summary.image("SmallerImg/y33", y33, max_outputs=1, collections=['Images'])

#     def generator(self, z, border, reuse, scope):
#         '''
#         X = down-sampled information
#         y = border information
#         '''
#         return generator_up(X=border, z=z, params=self.params['generator'], reuse=reuse, scope=scope)

#     def discriminator(self, X, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)


# class LapPatchWGANsimpleModel(GanModel):
#     def __init__(self, params, X, z, name='lapgansimple', is_3d=False):
#         ''' z must have the same dimension as X'''
#         super().__init__(params=params, name=name, is_3d=is_3d)
        
#         # A) Down sampling the image
#         self.upsampling = params['generator']['upsampling']

#         X0, border = tf.split(X, [1,3],axis=3)
#         self.Xs = down_sampler(X0, s=self.upsampling)

#         # The input is the downsampled image
#         inshape = self.Xs.shape.as_list()[1:]
#         self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')
#         # The border is a different input
#         inshape = border.shape.as_list()[1:]
#         self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
#         X1, X2, X3 = tf.split(self.border, [1,1,1],axis=3)
#         X1f = tf.reverse(X1, axis=[1])
#         X2f = tf.reverse(X2, axis=[2])
#         X3f = tf.reverse(X3, axis=[1,2])
#         flip_border = tf.concat([X1f, X2f, X3f], axis=3)
#         self.Xsu = up_sampler(self.y, s=self.upsampling)

#         self.G_fake = self.generator(X=self.Xsu, z=z, border=flip_border, reuse=False, scope='generator')


#         # E) Discriminator
#         self.D_real = self.discriminator(X0, self.Xsu, flip_border, reuse=False)
#         self.D_fake = self.discriminator(self.G_fake, self.Xsu, flip_border, reuse=True)

#         # F) Losses
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [self.G_fake, self.Xsu, flip_border], [X0, self.Xsu, flip_border])
#         #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
#         # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
#         # Min(D_loss_r - D_loss_f) = Min -D_loss_f
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f

#         # G) Summaries
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#         tf.summary.image("training/Input_Image", self.Xs, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Real_Diff", X0 - self.Xsu, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Fake_Diff", self.G_fake - self.Xsu, max_outputs=2, collections=['Images'])
#         if True:
#             # D) Concatenate back
#             top = tf.concat([X3,X2], axis=1)
#             bottom = tf.concat([X1,X0], axis=1)
#             bottom_g = tf.concat([X1,self.G_fake], axis=1)
#             full_img = tf.concat([top,bottom], axis=2)
#             full_img_fake = tf.concat([top,bottom_g], axis=2)
#             tf.summary.image("training/full_img_real", full_img, max_outputs=4, collections=['Images'])
#             tf.summary.image("training/full_img_fake", full_img_fake, max_outputs=4, collections=['Images'])
#             tf.summary.image("training/X0", X0, max_outputs=2, collections=['Images'])
#             tf.summary.image("training/X1", X1, max_outputs=1, collections=['Images'])
#             tf.summary.image("training/X2", X2, max_outputs=1, collections=['Images'])
#             tf.summary.image("training/X3", X3, max_outputs=1, collections=['Images'])
#             tf.summary.image("training/X1f", X1f, max_outputs=1, collections=['Images'])
#             tf.summary.image("training/X2f", X2f, max_outputs=1, collections=['Images'])
#             tf.summary.image("training/X3f", X3f, max_outputs=1, collections=['Images'])

#     def generator(self, X, z, border, reuse, scope):
#         return generator_up(tf.concat([X, border], axis=3), z, params=self.params['generator'], y=None, reuse=reuse, scope=scope)

#     def discriminator(self, X, Xsu, border, reuse):
#         return discriminator(tf.concat([X, Xsu, X-Xsu, border], axis=3), self.params['discriminator'], reuse=reuse)


# class LapPatchWGANsimpleUnfoldModel(GanModel):
#     def __init__(self, params, X, z, name='lapgansimpleunfold', is_3d=False):
#         ''' z must have the same dimension as X'''
#         super().__init__(params=params, name=name, is_3d=is_3d)
        
#         # A) Down sampling the image
#         self.upsampling = params['generator']['upsampling']

#         X0, border = tf.split(X, [1,3], axis=3)
#         self.Xs = down_sampler(X0, s=self.upsampling)

#         # The input is the downsampled image
#         inshape = self.Xs.shape.as_list()[1:]
#         self.y = tf.placeholder_with_default(self.Xs, shape=[None, *inshape], name='y')
       
#         # The border is a different input
#         inshape = border.shape.as_list()[1:]
#         self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
#         X1, X2, X3 = tf.split(self.border, [1,1,1],axis=3)
#         X1f = tf.reverse(X1, axis=[1])
#         X2f = tf.reverse(X2, axis=[2])
#         X3f = tf.reverse(X3, axis=[1,2])
#         flip_border = tf.concat([X1f, X2f, X3f], axis=3)

#         self.G_fake = self.generator(y=up_sampler(self.y, s=self.upsampling),
#                                      z=z,
#                                      border=flip_border,
#                                      reuse=False,
#                                      scope='generator')

#         # D) Concatenate back
#         top = tf.concat([X3,X2], axis=1)
#         bottom = tf.concat([X1,X0], axis=1)
#         bottom_g = tf.concat([X1,self.G_fake], axis=1)
#         X_real = tf.concat([top,bottom], axis=2)
#         G_fake = tf.concat([top,bottom_g], axis=2)
#         Xs_full = down_sampler(X_real, s=self.upsampling)
#         self.Xsu = up_sampler(Xs_full, s=self.upsampling)

#         # E) Discriminator
#         self.D_real = self.discriminator(X_real, self.Xsu, reuse=False)
#         self.D_fake = self.discriminator(G_fake, self.Xsu, reuse=True)

#         # F) Losses
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake, self.Xsu], [X_real, self.Xsu])
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f

#         # G) Summaries
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#         tf.summary.image("training/Real_full_image", X_real, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Fake_full_image", G_fake, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Downsample_X0", self.y, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Real_Diff", X_real - self.Xsu, max_outputs=1, collections=['Images'])
#         tf.summary.image("training/Fake_Diff", G_fake - self.Xsu, max_outputs=1, collections=['Images'])
#         if True:
#             tf.summary.image("checking/X0", X0, max_outputs=2, collections=['Images'])
#             tf.summary.image("checking/X1", X1, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X2", X2, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X3", X3, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X1f", X1f, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X2f", X2f, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X3f", X3f, max_outputs=1, collections=['Images'])

#     def generator(self, y, z, border, reuse, scope):
#         return generator_up(tf.concat([y, border], axis=3), z, self.params['generator'], y=None, reuse=reuse, scope=scope)

#     def discriminator(self, X, Xsu, reuse):
#         return discriminator(tf.concat([X, Xsu, X-Xsu], axis=3), self.params['discriminator'], reuse=reuse)


# class LapPatchWGANDirect(GanModel):
#     def __init__(self, params, X, z, name='lapgandirect', is_3d=False):
#         '''Some model for Ankit to try.
        
#         z must have the same dimension as X.
#         stride of 1
#         '''
#         super().__init__(params=params, name=name, is_3d=is_3d)
        
#         # A) Down sampling the image
#         self.upsampling = params['generator']['upsampling']

#         X0, border = tf.split(X, [1,3],axis=3)

#         # The border is a different input
#         inshape = border.shape.as_list()[1:]
#         self.border = tf.placeholder_with_default(border, shape=[None, *inshape], name='border')
#         X1, X2, X3 = tf.split(self.border, [1,1,1],axis=3)
#         X1f = tf.reverse(X1, axis=[1])
#         X2f = tf.reverse(X2, axis=[2])
#         X3f = tf.reverse(X3, axis=[1,2])
#         flip_border = tf.concat([X1f, X2f, X3f], axis=3)
#         self.G_fake = self.generator(y=flip_border, z=z, reuse=False, scope='generator')

#         # D) Concatenate back
#         top = tf.concat([X3,X2], axis=1)
#         bottom = tf.concat([X1,X0], axis=1)
#         bottom_g = tf.concat([X1,self.G_fake], axis=1)
#         X_real = tf.concat([top,bottom], axis=2)
#         G_fake = tf.concat([top,bottom_g], axis=2)

#         # E) Discriminator
#         self.D_real = self.discriminator(X_real, reuse=False)
#         self.D_fake = self.discriminator(G_fake, reuse=True)

#         # F) Losses
#         D_loss_f = tf.reduce_mean(self.D_fake)
#         D_loss_r = tf.reduce_mean(self.D_real)
#         gamma_gp = self.params['optimization']['gamma_gp']
#         D_gp = wgan_regularization(gamma_gp, self.discriminator, [G_fake], [X_real])
#         #D_gp = fisher_gan_regularization(self.D_real, self.D_fake, rho=gamma_gp)
#         # Max(D_loss_r - D_loss_f) = Min -(D_loss_r - D_loss_f)
#         # Min(D_loss_r - D_loss_f) = Min -D_loss_f
#         self._D_loss = -(D_loss_r - D_loss_f) + D_gp
#         self._G_loss = -D_loss_f

#         # G) Summaries
#         wgan_summaries(self._D_loss, self._G_loss, D_loss_f, D_loss_r)
#         tf.summary.image("training/Real_full_image", X_real, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Fake_full_image", G_fake, max_outputs=2, collections=['Images'])
#         tf.summary.image("training/Downsample_X0", self.y, max_outputs=2, collections=['Images'])
#         if True:
#             tf.summary.image("checking/X0", X0, max_outputs=2, collections=['Images'])
#             tf.summary.image("checking/X1", X1, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X2", X2, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X3", X3, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X1f", X1f, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X2f", X2f, max_outputs=1, collections=['Images'])
#             tf.summary.image("checking/X3f", X3f, max_outputs=1, collections=['Images'])

#     def generator(self, y, z, reuse, scope):
#         return generator_up(X, z, self.params['generator'], reuse=reuse, scope=scope)

#     def discriminator(self, X, Xsu, reuse):
#         return discriminator(X, self.params['discriminator'], reuse=reuse)



def js_gan_summaries(D_out_f, D_out_r):
    tf.summary.scalar("Disc/Out_f", tf.reduce_mean(D_out_f), collections=["Training"])
    tf.summary.scalar("Disc/Out_r", tf.reduce_mean(D_out_r), collections=["Training"])
    tf.summary.scalar("Disc/Out_f-r", tf.reduce_mean(D_out_f - D_out_r), collections=["Training"])

def wgan_summaries(D_loss, G_loss, D_loss_f, D_loss_r):
    tf.summary.scalar("Disc/Neg_Loss", -D_loss, collections=["Training"])
    tf.summary.scalar("Disc/Neg_Critic", D_loss_f - D_loss_r, collections=["Training"])
    tf.summary.scalar("Disc/Loss_f", D_loss_f, collections=["Training"])
    tf.summary.scalar("Disc/Loss_r", D_loss_r, collections=["Training"])
    tf.summary.scalar("Gen/Loss", G_loss, collections=["Training"])


def fisher_gan_regularization(D_real, D_fake, rho=1):
    with tf.variable_scope("discriminator", reuse=False):
        phi = tf.get_variable('lambda', shape=[],
            initializer=tf.initializers.constant(value=1.0, dtype=tf.float32))
        D_real2 = tf.reduce_mean(tf.square(D_real))
        D_fake2 = tf.reduce_mean(tf.square(D_fake))
        constraint = 1.0 - 0.5 * (D_real2 + D_fake2)

        # Here phi should be updated using another opotimization scheme
        reg_term = phi * constraint + 0.5 * rho * tf.square(constraint)
        print(D_real.shape)
        print(D_real2.shape)        
        print(constraint.shape)
        print(reg_term.shape)
    tf.summary.scalar("Disc/constraint", reg_term, collections=["Training"])
    tf.summary.scalar("Disc/reg_term", reg_term, collections=["Training"])
    return reg_term


def wgan_regularization(gamma, discriminator, list_fake, list_real):
    if not gamma:
        # I am not sure this part or the code is still useful
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
        D_gp = tf.constant(0, dtype=tf.float32)
        print(" [!] Using weight clipping")
    else:
        D_clip = tf.constant(0, dtype=tf.float32)
        # calculate `x_hat`
        assert(len(list_fake) == len(list_real))
        bs = tf.shape(list_fake[0])[0]
        eps = tf.random_uniform(shape=[bs], minval=0, maxval=1)

        x_hat = []
        for fake, real in zip(list_fake, list_real):
            singledim = [1]* (len(fake.shape.as_list())-1)
            eps = tf.reshape(eps, shape=[bs,*singledim])
            x_hat.append(eps * real + (1.0 - eps) * fake)

        D_x_hat = discriminator(*x_hat, reuse=True)

        # gradient penalty
        gradients = tf.gradients(D_x_hat, x_hat)
        norm_gradient_pen = tf.norm(gradients[0], ord=2)
        D_gp = gamma * tf.square(norm_gradient_pen - 1.0)
        tf.summary.scalar("Disc/GradPen", D_gp, collections=["Training"])
        tf.summary.scalar("Disc/NormGradientPen", norm_gradient_pen, collections=["Training"])
    return D_gp


#Roth et al. 2017, see https://github.com/rothk/Stabilizing_GANs
def js_regularization(D1_logits, D1_arg, D2_logits, D2_arg, batch_size):
    #print("In shapes: {}, {}, {}, {}".format(D1_logits.shape, D1_arg.shape, D2_logits.shape, D2_arg.shape))
    D1 = tf.nn.sigmoid(D1_logits)
    D2 = tf.nn.sigmoid(D2_logits)
    bs = tf.shape(D1)[0]
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    #print(grad_D1_logits.shape)
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [bs,-1]), axis=1, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [bs,-1]), axis=1, keep_dims=True)

    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    print("Shapes: {}=?={} and {}=?={}".format(grad_D1_logits_norm.shape, D1.shape,
                                               grad_D2_logits_norm.shape, D2.shape))
    #assert grad_D1_logits_norm.shape == D1.shape
    #assert grad_D2_logits_norm.shape == D2.shape

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer


def get_conv(data_size):
    if data_size == 3:
        return conv3d
    elif data_size == 2:
        return conv2d
    elif data_size == 1:
        return conv1d
    else:
        raise ValueError("Wrong data_size")


def deconv(in_tensor, bs, sx, n_filters, shape, stride, summary, conv_num, use_spectral_norm, sy=None, sz=None, data_size=2):
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    if data_size==3:
        output_shape = [bs, sx, sy, sz, n_filters]
        out_tensor = deconv3d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_3d'.format(conv_num),
                              use_spectral_norm=use_spectral_norm,
                              summary=summary)
    elif data_size==2:
        output_shape = [bs, sx, sy, n_filters]
        out_tensor = deconv2d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_2d'.format(conv_num),
                              use_spectral_norm=use_spectral_norm,
                              summary=summary)
    elif data_size==1:
        output_shape = [bs, sx, n_filters]
        out_tensor = deconv1d(in_tensor,
                              output_shape=output_shape,
                              shape=shape,
                              stride=stride,
                              name='{}_deconv_1d'.format(conv_num),
                              use_spectral_norm=use_spectral_norm,
                              summary=summary)
    else:
        raise ValueError("Wrong data_size")

    return out_tensor


def apply_non_lin(non_lin, x, reuse):
    if non_lin:
        if type(non_lin)==str:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(non_lin), reuse)
        else:
            x = non_lin(x)   
            rprint('    Costum non linearity: {}'.format(non_lin), reuse)

    return x


def legacy_cdf_block(x, params, reuse):
    cdf = tf_cdf(x, params['cdf'])
    rprint('    Cdf layer: {}'.format(params['cdf']), reuse)
    rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    if params['channel_cdf']:
        lst = []
        for i in range(x.shape[-1]):
            lst.append(tf_cdf(x, params['channel_cdf'],
                              name="cdf_weight_channel_{}".format(i)))
            rprint('        Channel Cdf layer: {}'.format(params['cdf']), reuse)
        lst.append(cdf)
        cdf = tf.concat(lst, axis=1)
        rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    cdf = linear(cdf, 2 * params['cdf'], 'cdf_full', summary=params['summary'])
    cdf = params['activation'](cdf)
    rprint('     CDF Full layer with {} outputs'.format(2 * params['cdf']), reuse)
    rprint('         Size of the CDF variables: {}'.format(cdf.shape), reuse)
    return cdf


def cdf_block(x, params, reuse):
    assert ('cdf_block' in params.keys())
    block_params = params['cdf_block']
    assert ('cdf_in' in block_params.keys() or 'channel_cdf' in block_params.keys())
    use_first = block_params.get('use_first_channel', False)
    cdf = None
    if block_params.get('cdf_in', None):
        cdf = tf_cdf(x, block_params['cdf_in'], use_first_channel=use_first)
        rprint('    Cdf layer: {}'.format(block_params['cdf_in']), reuse)
        rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    if block_params.get('channel_cdf', None):
        lst = []
        for i in range(x.shape[-1]):
            lst.append(tf_cdf(x[:,:,:,i], block_params['channel_cdf'], use_first_channel=False,
                              name="cdf_weight_channel_{}".format(i)))
            rprint('        Channel Cdf layer: {}'.format(block_params['channel_cdf']), reuse)
        if block_params.get('cdf_in', None):
            lst.append(cdf)
        cdf = tf.concat(lst, axis=1)
        rprint('         Size of the cdf variables: {}'.format(cdf.shape), reuse)
    out_dim = block_params.get('cdf_out', 2 * block_params.get('cdf_in',8))
    cdf = linear(cdf, out_dim, 'cdf_full', summary=params['summary'])
    cdf = params['activation'](cdf)
    rprint('     CDF Full layer with {} outputs'.format(out_dim), reuse)
    rprint('         Size of the CDF variables: {}'.format(cdf.shape), reuse)
    return cdf


def histogram_block(x, params, reuse):
    hist = learned_histogram(x, params['histogram'])
    out_dim = params['histogram'].get('full', 32)
    hist = linear(hist, out_dim, 'hist_full', summary=params['summary'])
    hist = params['activation'](hist)
    rprint('     Histogram full layer with {} outputs'.format(out_dim), reuse)
    rprint('         Size of the histogram variables: {}'.format(hist.shape), reuse)
    return hist


def discriminator(x, params, z=None, reuse=True, scope="discriminator"):
    conv = get_conv(params['data_size'])

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    for it, st in enumerate(params['stride']):
        if not(isinstance(st ,list) or isinstance(st ,tuple)):
            params['stride'][it] = [st]*params['data_size']


    with tf.variable_scope(scope, reuse=reuse):
        rprint('Discriminator \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)
        if params['non_lin']:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)


        if params['cdf']:
            cdf = legacy_cdf_block(x, params, reuse)
        if params['cdf_block']:
            assert(not params['cdf'])
            cdf = cdf_block(x, params, reuse)
        if params.get('histogram', None):
            print('generating histogram block')
            hist = histogram_block(x, params, reuse)
        if params['moment']:
            rprint('    Covariance layer with {} shape'.format(params['moment']), reuse)
            cov = tf_covmat(x, params['moment'])
            rprint('        Layer output {} shape'.format(cov.shape), reuse)
            cov = reshape2d(cov)
            rprint('        Reshape output {} shape'.format(cov.shape), reuse)
            nel = np.prod(params['moment'])**2
            cov = linear(cov, nel, 'cov_full', summary=params['summary'])
            cov = params['activation'](cov)
            rprint('     Covariance Full layer with {} outputs'.format(nel), reuse)
            rprint('         Size of the CDF variables: {}'.format(cov.shape), reuse)

        for i in range(nconv):
            # TODO: this really needs to be cleaned uy...

            if params['data_size']==1 and not(i==0):
                if params.get('apply_phaseshuffle', False):
                    rprint('     Phase shuffle', reuse)               
                    x=apply_phaseshuffle(x)
            if params['inception']:
                x = inception_conv(in_tensor=x, 
                                    n_filters=params['nfilter'][i], 
                                    stride=params['stride'][i], 
                                    summary=params['summary'], 
                                    num=i,
                                    data_size=params['data_size'],
                                    use_spectral_norm=params['spectral_norm'],
                                    merge=(i == (nconv-1))
                                    )
                rprint('     {} Inception(1x1,3x3,5x5) layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            elif params.get('separate_first', False) and i == 0:
                n_out = params['nfilter'][i] // (int(x.shape[3]) + 1)
                lst = []
                for j in range(x.shape[3]):
                    lst.append(conv(x[:,:,:,j:j+1],
                        nf_out=n_out,
                        shape=params['shape'][i],
                        stride=params['stride'][i],
                        use_spectral_norm=params['spectral_norm'],
                        name='{}_conv{}'.format(i,j),
                        summary=params['summary']))
                lst.append(conv(x[:,:,:,:],
                        nf_out=params['nfilter'][i] - (n_out * int(x.shape[3])),
                        shape=params['shape'][i],
                        stride=params['stride'][i],
                        use_spectral_norm=params['spectral_norm'],
                        name='{}_conv_full'.format(i),
                        summary=params['summary']))
                x = tf.concat(lst, axis=3)
            else:
                x = conv(x,
                         nf_out=params['nfilter'][i],
                         shape=params['shape'][i],
                         stride=params['stride'][i],
                         use_spectral_norm=params['spectral_norm'],
                         name='{}_conv'.format(i),
                         summary=params['summary'])
                rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)

            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = params['activation'](x)

        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)

        if z is not None:
            x = tf.concat([x, z], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        if params['cdf'] or params['cdf_block']:
            x = tf.concat([x, cdf], axis=1)
            rprint('     Contenate with CDF variables to {}'.format(x.shape), reuse)
        if params.get('histogram', None):
            x = tf.concat([x, hist], axis=1)
            rprint('     Contenate with Histogram variables to {}'.format(x.shape), reuse)
        if params['moment']:
            x = tf.concat([x, cov], axis=1)
            rprint('     Contenate with covairance variables to {}'.format(x.shape), reuse)           

        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = params['activation'](x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)
        if params['minibatch_reg']:
            rprint('     Minibatch regularization', reuse)
            x = mini_batch_reg(x, n_kernels=150, dim_per_kernel=30)
            rprint('       Size of the variables: {}'.format(x.shape), reuse)

        x = linear(x, 1, 'out', summary=params['summary'])
        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x

def generator(x, params, X=None, y=None, reuse=True, scope="generator"):
    assert(len(params['stride']) == len(params['nfilter'])
           == len(params['batch_norm'])+1)
    nconv = len(params['stride'])
    nfull = len(params['full'])
    for it, st in enumerate(params['stride']):
        if not(isinstance(st ,list) or isinstance(st ,tuple)):
            params['stride'][it] = [st]*params['data_size']


    with tf.variable_scope(scope, reuse=reuse):
        rprint('Generator \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        if y is not None:
            x = tf.concat([x, y], axis=1)
            rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i),
                       summary=params['summary'])
            x = params['activation'](x)
            rprint('     {} Full layer with {} outputs'.format(i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        bs = tf.shape(x)[0]  # Batch size

        # The following code can probably be much more beautiful.
        if params['data_size']==3:
            # nb pixel
            # if params.get('in_conv_shape', None) is not None:
            sx, sy, sz = params['in_conv_shape']
            # else:
            #     if X is not None:
            #         sx, sy, sz = X.shape.as_list()[1:4]
            #     else:
            #         sx = np.int(np.round((np.prod(x.shape.as_list()[1:]))**(1/3)))
            #         sy, sz = sx, sx
            c = np.int(np.round(np.prod(x.shape.as_list()[1:])))//(sx*sy*sz)
            x = tf.reshape(x, [bs, sx, sy, sz, c], name='vec2img')
            rprint('     Reshape to {}'.format(x.shape), reuse)
            if X is not None:
                x = tf.concat([x, X], axis=4)
                rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        elif params['data_size']==2:
            # if params.get('in_conv_shape', None) is not None:
            sx, sy = params['in_conv_shape']
            sz = None
            # else:
            #     # nb pixel
            #     if X is not None:
            #         sx, sy = X.shape.as_list()[1:3]
            #     else:
            #         sx = np.int(np.round(
            #             np.sqrt(np.prod(x.shape.as_list()[1:]))))
            #         sy = sx
            c = np.int(np.round(np.prod(x.shape.as_list()[1:])))//(sx*sy)
            x = tf.reshape(x, [bs, sx, sy, c], name='vec2img')
            rprint('     Reshape to {}'.format(x.shape), reuse)
            if X is not None:
                x = tf.concat([x, X], axis=3)
                rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)
        else:
            # if params.get('in_conv_shape', None) is not None:
            sx = params['in_conv_shape'][0]
            sy, sz = None, None
            # else:
            #     if X is not None:
            #         sx = X.shape.as_list()[1]
            #     else:
            #         sx = np.int(np.round(np.prod(x.shape.as_list()[1:])))
            c = np.int(np.round(np.prod(x.shape.as_list()[1:])))//sx
            x = tf.reshape(x, [bs, sx, c], name='vec2img')
            rprint('     Reshape to {}'.format(x.shape), reuse)

            if X is not None:
                x = tf.concat([x, X], axis=2)
                rprint('     Contenate with latent variables to {}'.format(x.shape), reuse)

        
        if params.get('use_conv_over_deconv', True):
            conv_over_deconv = stride2reduction(params['stride'])==1 # If true use conv, else deconv
        else:
            conv_over_deconv = False

        for i in range(nconv):
            sx = sx * params['stride'][i][0]
            if params['data_size']>1:
                sy = sy * params['stride'][i][1]
            if params['data_size']>2:
                sz = sz * params['stride'][i][2]
            if params['residual'] and (i%2 != 0) and (i < nconv-2): # save odd layer inputs for residual connections
                residue = x

            if params['inception']:
                if conv_over_deconv:
                    x = inception_conv(in_tensor=x, 
                                    n_filters=params['nfilter'][i], 
                                    stride=params['stride'][i], 
                                    summary=params['summary'], 
                                    num=i,
                                    data_size=params['data_size'], 
                                    use_spectral_norm=params['spectral_norm'],
                                    merge= (True if params['residual'] else (i == (nconv-1)) )
                                    )
                    rprint('     {} Inception conv(1x1,3x3,5x5) layer with {} channels'.format(i, params['nfilter'][i]), reuse)

                else:
                    x = inception_deconv(in_tensor=x, 
                                        bs=bs, 
                                        sx=sx, 
                                        n_filters=params['nfilter'][i], 
                                        stride=params['stride'][i], 
                                        summary=params['summary'], 
                                        num=i, 
                                        data_size=params['data_size'],
                                        use_spectral_norm=params['spectral_norm'],
                                        merge= (True if params['residual'] else (i == (nconv-1)) )
                                        )
                    rprint('     {} Inception deconv(1x1,3x3,5x5) layer with {} channels'.format(i, params['nfilter'][i]), reuse)

            else:       
                x = deconv(in_tensor=x, 
                           bs=bs, 
                           sx=sx,
                           n_filters=params['nfilter'][i],
                           shape=params['shape'][i],
                           stride=params['stride'][i],
                           summary=params['summary'],
                           conv_num=i,
                           use_spectral_norm=params['spectral_norm'],
                           sy = sy,
                           sz = sz,
                           data_size=params['data_size']
                           )
                rprint('     {} Deconv layer with {} channels'.format(i+nfull, params['nfilter'][i]), reuse)
            # residual connections before ReLU of every even layer, except 0th and last layer
            if params['residual'] and (i != 0) and (i != nconv-1) and (i%2 == 0):
                x = x + residue
                rprint('         Residual connection', reuse)


            if i < nconv-1:
                if params['batch_norm'][i]:
                    x = batch_norm(x, name='{}_bn'.format(i), train=True)
                    rprint('         Batch norm', reuse)

                x = params['activation'](x)
                rprint('         Non linearity applied', reuse)

            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        if len(params['one_pixel_mapping']):
            x = one_pixel_mapping(x,
                                  params['one_pixel_mapping'],
                                  summary=params['summary'],
                                  reuse=reuse)

        x = apply_non_lin(params['non_lin'], x, reuse)

        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def encoder(x, params, latent_dim, reuse=True, scope="encoder"):

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Encoder \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        for i in range(nconv):
            x = conv2d(x,
                       nf_out=params['nfilter'][i],
                       shape=params['shape'][i],
                       stride=params['stride'][i],
                       name='{}_conv'.format(i),
                       summary=params['summary'])
            rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)
            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = lrelu(x)

        x = conv2d(x,
                   nf_out=64,
                   shape=[1,1],
                   stride=1,
                   name='out',
                   summary=params['summary'])
        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)
        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = lrelu(x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

        #x = linear(x, latent_dim, 'out', summary=params['summary'])

        # x = tf.sigmoid(x)
        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)
    return x


def one_pixel_mapping(x, n_filters, summary=True, reuse=False):
    """One pixel mapping."""
    rprint('  Begining of one Pixel Mapping '+''.join(['-']*20), reuse)
    xsh = tf.shape(x) 

    rprint('     The input is of size {}'.format(x.shape), reuse)
    x = tf.reshape(x, [xsh[0], prod(x.shape.as_list()[1:]), 1, 1])
    rprint('     Reshape x to size {}'.format(x.shape), reuse)
    nconv = len(n_filters)
    for i, n_filter in enumerate(n_filters):
        x = conv2d(x,
                   nf_out=n_filter,
                   shape=[1, 1],
                   stride=1,
                   name='{}_1x1conv'.format(i),
                   summary=summary)

        rprint('     {} 1x1 Conv layer with {} channels'.format(i, n_filter), reuse)    
        x = lrelu(x)
        rprint('         Size of the variables: {}'.format(x.shape), reuse)

    x = conv2d(x,
               nf_out=1,
               shape=[1, 1],
               stride=1,
               name='final_1x1conv',
               summary=summary)
    x = tf.reshape(x, xsh)
    rprint('     Reshape x to size {}'.format(x.shape), reuse)
    rprint('  End of one Pixel Mapping '+''.join(['-']*20)+'\n', reuse)
    return x

def generator_border(x, params, X=None, y=None, reuse=True, scope="generator"):
    params_border = params['borders']
    conv = get_conv(params_border['data_size'])

    assert(len(params_border['stride']) == len(params_border['nfilter'])
           == len(params_border['batch_norm']))
    nconv_border = len(params_border['stride'])
    with tf.variable_scope(scope, reuse=reuse):
        rprint('Border block', reuse)
        rprint('\n'+(''.join(['-']*50)), reuse)
        rprint('     BORDER:  The input is of size {}'.format(y.shape), reuse)
        imgt = y
        for i in range(nconv_border):
            imgt = conv(imgt,
                       nf_out=params_border['nfilter'][i],
                       shape=params_border['shape'][i],
                       stride=params_border['stride'][i],
                       name='{}_conv'.format(i),
                       summary=params['summary'])
            rprint('     BORDER: {} Conv layer with {} channels'.format(i, params_border['nfilter'][i]), reuse)
            if params_border['batch_norm'][i]:
                imgt = batch_norm(imgt, name='{}_border_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         BORDER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        imgt = reshape2d(imgt, name='border_conv2vec')
        
        wf = params_border['width_full']
        if wf is not None:
            st = y.shape.as_list()
            if params_border['data_size']==1:
                # We take the begining or the signal as it is flipped.
                border = reshape2d(tf.slice(y, [0, 0, 0], [-1, wf, st[2]]), name='border2vec')
            elif params_border['data_size']==2:
                print('Warning slicing only on side')
                # This is done for the model inpaintingGAN that is supposed to work with spectrograms...
                # We take the begining or the signal as it is flipped.
                border = reshape2d(tf.slice(y, [0, 0, 0, 0], [-1, wf, st[2], -1]), name='border2vec')
                # border = reshape2d(tf.slice(img, [0, st[1]-wf, 0, 0], [-1, wf, st[2], st[3]]), name='border2vec')
            elif params_border['data_size']==3:
                raise NotImplementedError()
            else:
                raise ValueError('Incorrect data_size')
            rprint('     BORDER:  Size of the border variables: {}'.format(border.shape), reuse)
            # rprint('     Latent:  Size of the Z variables: {}'.format(x.shape), reuse)
            y = tf.concat([imgt, border], axis=1)
        else:
            y = imgt

        rprint('     BORDER:  Size of the conv variables: {}'.format(imgt.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)

        return generator(x, params, X=X, y=y, reuse=reuse, scope=scope)

def stride2reduction(stride):
    # This code works with array and single element in stride
    reduction = 1
    for st in stride:
        reduction *= np.array([st]).flatten()[0]
    return reduction