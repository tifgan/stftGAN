# stftGAN

TODO: Add intro + nice image

**paper**: [Adversarial Generation of Time-Frequency Features
with application in audio synthesis][paper]

**website**: [Webpage with supporting material][website]

[paper]: https://arxiv.org/abs/...........
[website]: https://tifgan.github.io/

#### Abstract
Time-frequency (TF) representations provide powerful and intuitive features for the analysis of time series such as audio. But still, generative modeling of audio in the TF domain is a subtle matter. Consequently, neural audio synthesis widely relies on directly modeling the waveform and previous attempts at unconditionally synthesizing audio from neurally generated TF features still struggle to produce audio at satisfying quality. In this contribution, focusing on the short-time Fourier transform, we discuss the challenges that arise in audio synthesis based on generated TF features and how to overcome them. We demonstrate the potential of deliberate generative TF modeling by training a generative adversarial network (GAN) on short-time Fourier features. We show that our TF-based network was able to outperform the state-of-the-art GAN generating waveform, despite the similar architecture in the two networks. 


## Installation

First clone the repository.

```
git clone https://github.com/tifgan/stftGAN.git 
```

#### Softwares requirements
While most of the code is written in Python (we used version 3.5), the phase recovery part requires the use of `octave` or `MATLAB`. We are currently working to provide a full Python implementation.

#### Ltfatpy requirements

`ltfatpy`, one of the package, requires the installation of `fftw3` and `lapack`. Please check the page
http://dev.pages.lis-lab.fr/ltfatpy/install.html for a proper installation.

Alternatively, on Debian based linux, you may try:
```
apt install libfftw3-dev liblapack-dev
```

For macOS based systems, you may try:
```
brew install fftw
```


#### Python requirements

*We highly recommend to work in a virtual environnement.*

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

Once the speech commands dataset is generated following the notebook, any of the files inside of `specgan/train_commands` can be run to train a TiFGAN.

For example TiFGAN-M can be trained using:
```
TODO: add the python line
```

## Generate samples

Afterwards, the corresponding file in `specgan/generate_commands` will generate 256 samples from the last checkpoint. Phases need to be recovered using the code available at `phase_recovery`. We are developing an implementation of PGHI on python.

To generate the magnitudes from TiFGAN-M , please use:
```
TODO: add the python line
```
Then, the signals can be reconstructed using:
```
TODO: add the octave line
```

## Pre-trained networks
The checkpoints used for the evaluation of the [paper][paper] can be downloaded [here][linkcheckpoint]. Please extract the archiv in the folder `TODO: add the folder`. To generate magnitudes using those checkpoints, use the following command:
```
TODO: add the python line
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
