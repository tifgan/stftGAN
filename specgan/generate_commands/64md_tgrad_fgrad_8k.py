import sys
sys.path.insert(0, '../../')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import SpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools.data import fmap
import functools
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.io

downscale = 1

print('start')

mat_path = "../../data/test_spectrograms_and_derivs_1.mat"
raw_data = scipy.io.loadmat(mat_path)
print(raw_data['logspecs'].shape)
print(raw_data['tgrad'].shape)
preprocessed_images = raw_data['logspecs']
tgrads = raw_data['tgrad']
fgrads = raw_data['fgrad']


del raw_data
print(preprocessed_images.shape)
print(tgrads.shape)
print(np.max(preprocessed_images[:, :256, :]))
print(np.min(preprocessed_images[:, :256, :]))
print(np.mean(preprocessed_images[:, :256, :]))

tgrads[np.isnan(tgrads)] = 0
tgrads = np.clip(tgrads, -1, 1)

fgrads[np.isnan(fgrads)] = 0
fgrads = np.clip(fgrads, -1, 1)

print(np.max(tgrads[:, :256, :]))
print(np.min(tgrads[:, :256, :]))
print(np.mean(tgrads[:, :256, :]))

print(np.max(fgrads[:, :256, :]))
print(np.min(fgrads[:, :256, :]))
print(np.mean(fgrads[:, :256, :]))

dataset = Dataset(np.stack([preprocessed_images[:, :256], tgrads[:, :256], fgrads[:, :256]], axis=-1))

time_str = 'commands_md64_tgrads_fgrads_8k'
global_path = '../../saved_results'

name = time_str

from gantools import blocks
bn = False

md = 64

params_discriminator = dict()
params_discriminator['stride'] = [2,2,2,2,2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
params_discriminator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2
params_discriminator['apply_phaseshuffle'] = True
params_discriminator['spectral_norm'] = True
params_discriminator['activation'] = blocks.lrelu


params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2]
params_generator['latent_dim'] = 100
params_generator['consistency_contribution'] = 0
params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 3]
params_generator['shape'] = [[12, 3],[12, 3], [12, 3],[12, 3],[12, 3]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.tanh
params_generator['activation'] = tf.nn.relu
params_generator['data_size'] = 2
params_generator['spectral_norm'] = True 
params_generator['in_conv_shape'] =[8, 4]

params_optimization = dict()
params_optimization['batch_size'] = 64
params_optimization['epoch'] = 10000
params_optimization['n_critic'] = 5
params_optimization['generator'] = dict()
params_optimization['generator']['optimizer'] = 'adam'
params_optimization['generator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['generator']['learning_rate'] = 1e-4
params_optimization['discriminator'] = dict()
params_optimization['discriminator']['optimizer'] = 'adam'
params_optimization['discriminator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['discriminator']['learning_rate'] = 1e-4



# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = [256, 128, 3] # Shape of the image
params['net']['gamma_gp'] = 10 # Gradient penalty
params['net']['fs'] = 16000//downscale
params['net']['loss_type'] ='wasserstein'

params['optimization'] = params_optimization
params['summary_every'] = 100 # Tensorboard summaries every ** iterations
params['print_every'] = 50 # Console summaries every ** iterations
params['save_every'] = 1000 # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 500

resume, params = utils.test_resume(True, params)
params['optimization']['epoch'] = 10000

from gantools.plot import colorize
           
class ModSpectrogramGAN(SpectrogramGAN):
    def _build_net(self):
        super(SpectrogramGAN, self)._build_net()
        consistency_contribution = self._params['generator']['consistency_contribution']
        print("consistency_contribution", consistency_contribution)
        self._R_Con = self.consistency((self.X_real[:,:,:,0]-1)*5)
        self._F_Con = self.consistency((self.X_fake[:,:,:,0]-1)*5)
        self._mean_R_Con, self._std_R_Con = tf.nn.moments(self._R_Con, axes=[0])
        self._mean_F_Con, self._std_F_Con = tf.nn.moments(self._F_Con, axes=[0])
                        
        self._G_Reg =  tf.abs(self._mean_R_Con - self._mean_F_Con)
        self._G_loss += consistency_contribution*self._G_Reg
    
    def _build_image_summary(self):
        vmin = tf.reduce_min(self.X_real)
        vmax = tf.reduce_max(self.X_real)
        X_real = self.X_real[:,:,:,0]
        X_fake = self.X_fake[:,:,:,0]

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
            
wgan = GANsystem(ModSpectrogramGAN, params)

nsamples = 256
nlatent = 100

def clip_dist2(nsamples, nlatent, m=2.5):
    shape = [nsamples, nlatent]
    z = np.random.randn(*shape)
    support = np.logical_or(z<-m, z>m)
    while np.sum(support):
        z[support] = np.random.randn(*shape)[support]
        support = np.logical_or(z<-m, z>m)
    return z

d2 = clip_dist2(nsamples, nlatent)
np.max(d2), np.min(d2)

generated_signals = np.squeeze(wgan.generate(N=nsamples, z=d2))
generated_signals = np.concatenate([generated_signals, np.zeros_like(generated_signals)[:, 0:1, :, :]], axis=1) #Fill last column of freqs with zeros
generated_spectrograms = 5*(generated_signals[:, :, :, 0]-1) # Undo preprocessing
generated_tgrads = 64*generated_signals[:, :, :, 1] # Undo preprocessing
generated_fgrads = 256*generated_signals[:, :, :, 2] # Undo preprocessing

##Phase recovery

from data.ourLTFATStft import LTFATStft

fft_hop_size = 128 # Change the fft params if the dataset was generated with others
fft_window_length = 512
L = 16384
clipBelow = -10

anStftWrapper = LTFATStft()
from phase_recovery.numba_pghi import pghi

generated_tgrads = 2*np.pi*fft_hop_size/L * generated_tgrads #Prescale
generated_fgrads = - 2*np.pi/fft_window_length * (generated_fgrads + np.tile(np.arange(128)*fft_hop_size, (int(fft_window_length/2+1), 1))) #Prescale

reconstructed_audios = np.zeros([len(generated_signals), L])
for index, logMagSpectrogram in enumerate(generated_spectrograms):
    logMagSpectrogram = logMagSpectrogram.astype(np.float64)
    phase = pghi(logMagSpectrogram, generated_tgrads[index], generated_fgrads[index], fft_hop_size, fft_window_length, L, tol=10)
    reconstructed_audios[index] = anStftWrapper.reconstructSignalFromLoggedSpectogram(logMagSpectrogram, phase, windowLength=fft_window_length, hopSize=fft_hop_size)
    print(reconstructed_audios[index].max())

print("reconstructed audios!")

scipy.io.savemat('commands_listen.mat', {"reconstructed": reconstructed_audios, "generated_spectrograms": generated_signals})