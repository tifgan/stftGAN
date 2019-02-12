# TiFGAN: Time Frequency Generative Adversarial Networks

This repository contains the code accompanying the paper [Adversarial Generation of Time-Frequency Features with application in audio synthesis][paper]. Supplementary material can be found at [this webpage.][website]

[paper]: https://arxiv.org/abs/...........
[website]: https://tifgan.github.io/

#### Abstract
Time-frequency (TF) representations provide powerful and intuitive features for the analysis of time series such as audio. But still, generative modeling of audio in the TF domain is a subtle matter. Consequently, neural audio synthesis widely relies on directly modeling the waveform and previous attempts at unconditionally synthesizing audio from neurally generated TF features still struggle to produce audio at satisfying quality. In this contribution, focusing on the short-time Fourier transform, we discuss the challenges that arise in audio synthesis based on generated TF features and how to overcome them. We demonstrate the potential of deliberate generative TF modeling by training a generative adversarial network (GAN) on short-time Fourier features. We show that our TF-based network was able to outperform the state-of-the-art GAN generating waveform, despite the similar architecture in the two networks. 


## Installation

The easiest way to access the code is to clone the repository:

```
git clone https://github.com/tifgan/stftGAN.git 
cd stftGAN
```

#### Software requirements
While most of the code is written in Python (we used version 3.5), the phase recovery part requires the use of O`ctave` or `MATLAB`. We are currently working to provide a full-Python implementation. Unfortunately, for now, you need to install one of these two software.

You also need to install the [LTFAT][ltfat.github.io] library a be sure that the base function ltfatstart is in the accessible path MATLAB/octave.

#### Ltfatpy requirements

`ltfatpy`, one of the packages, requires the installation of `fftw3` and `lapack`. Please check the page
http://dev.pages.lis-lab.fr/ltfatpy/install.html for a proper installation.

Alternatively, on Debian based linux, you may try:
```
apt install libfftw3-dev liblapack-dev
```

For macOS based systems, you may try:
```
brew install fftw lapack
```


#### Python requirements

*We highly recommend working in a virtual environment.*

You can install the required packages with the following command:
```
pip install -r requirements.txt
```


## Datasets

Here are some datasets we used to train TifGAN:

- [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz)
- [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz)
- [Drum sound effects](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz)

The data should be extracted in the  `data` folder. On the notebook inside the folder there are instructions to generate a dataset from audio files.

## Train a TiFGAN

Once the speech commands dataset is generated following the notebook, any of the files inside of `specgan/train_commands` can be used to train a TiFGAN.

For example, TiFGAN-M can be trained using:
```
specgan/train_commands
python 64md_8k.py
```

## Generate samples

Afterwards, the corresponding file in `specgan/generate_commands` will generate 256 samples from the last checkpoint. Phases need to be recovered using the code available at `phase_recovery`. We are developing an implementation of PGHI on python.

To generate the magnitudes from TiFGAN-M , please use:
```
cd specgan/generate_commands
python 64md_8k.py
```
Then, the signals can be reconstructed in MATLAB/octave with the following scripts `recover_phase_from_mags.m` or `recover_phase_from_mags_and_derivs.m`. Alternatively, for MATLAB you can try the following one-liner command:
```
matlab -nodesktop -nosplash -nodisplay -r \
"try, run('recover_phase_from_mags.m'), catch, exit(1), end, exit(0);"
```
This command will work only if the function ltfatstart is in the path of MATLAB/octave.

## Pre-trained networks
The checkpoints used for the evaluation of the [paper][paper] can be downloaded [here][https://zenodo.org/record/2562819]. Please extract the archiv in the folder `saved_results`. To generate magnitudes using those checkpoints, use on of the following commands:
```
cd specgan/generate_commands
python 64md_8k.py 
```
or
```
cd specgan/generate_commands
python 64md_tgrad_fgrad_8k.py 
```
or
```
cd specgan/generate_piano
python 8k_tall_large.py  
```


[linkcheckpoint]: https://...

## Evaluation

The Inception Score (IS) is computed with [score.py](https://github.com/chrisdonahue/wavegan/blob/master/eval/inception/score.py) from the WaveGAN repository. To also compute the Fréchet Inception Distance (FID), we submitted an extension of that file which is currently being processed as a [pull request](https://github.com/chrisdonahue/wavegan/pull/23). 

## License & co

The content of this repository is released under the terms of the [GNU3 license](LICENCE.txt).
Please cite our [paper] if you use it.

```
@article{,
  title = {Adversarial Generation of Time-Frequency Features
with application in audio synthesis},
  author = {Marafioti, Andrès and Holighaus, Nicki and Perraudin, Nathanaël and Majdak, Piotr},
  journal = {arXiv},
  year = {2019},
  url = {https://arxiv.org/abs/.......},
}
```
