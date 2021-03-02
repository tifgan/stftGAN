# TiFGAN: Time Frequency Generative Adversarial Networks

This repository contains the code accompanying the paper [Adversarial Generation of Time-Frequency Features with application in audio synthesis][paper]. Supplementary material can be found at [this webpage.][website]

[paper]: https://arxiv.org/abs/1902.04072
[website]: https://tifgan.github.io/

#### Abstract
Time-frequency (TF) representations provide powerful and intuitive features for the analysis of time series such as audio. But still, generative modeling of audio in the TF domain is a subtle matter. Consequently, neural audio synthesis widely relies on directly modeling the waveform and previous attempts at unconditionally synthesizing audio from neurally generated TF features still struggle to produce audio at satisfying quality. In this contribution, focusing on the short-time Fourier transform, we discuss the challenges that arise in audio synthesis based on generated TF features and how to overcome them. We demonstrate the potential of deliberate generative TF modeling by training a generative adversarial network (GAN) on short-time Fourier features. We show that our TF-based network was able to outperform the state-of-the-art GAN generating waveform, despite the similar architecture in the two networks. 


## Installation

The easiest way to access the code is to clone the repository:

```
git clone https://github.com/tifgan/stftGAN.git 
cd stftGAN
```

#### Python requirements

*We highly recommend working in a virtual environment.*

You can install the required packages with the following command:
```
pip install -r requirements.txt
```

#### Ltfatpy requirements

`ltfatpy`, one of the packages, requires the installation of `fftw3` and `lapack`. Please check [this page](http://dev.pages.lis-lab.fr/ltfatpy/install.html) for a proper installation.

Alternatively, on Debian based linux, you may try:
```
apt install libfftw3-dev liblapack-dev
```

For macOS based systems, you may try:
```
brew install fftw lapack
```

## Datasets

To download the commands dataset used on the paper and pre-process the Time-Frequency features used for training the networks, run on the main folder:
```
python download_commands.py
python generate_commands_dataset.py
```

Here are some datasets we used to train TifGAN:

- [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz)
- [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz)
- [Drum sound effects](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz)

To work with these or other dataset, they just follow the notebook inside the `data` folder. It contains instructions to generate a dataset from a set of audio files.

## Train a TiFGAN

Once the speech commands dataset is generated following the notebook, any of the files inside of `specgan/train_commands` can be used to train a TiFGAN.

For example, TiFGAN-M can be trained using:
```
cd specgan/train_commands
python 64md_8k.py
```

## Generate samples

Afterwards, the corresponding file in `specgan/generate_commands` will generate 256 samples from the last checkpoint. Phases need to be recovered using the code available at `phase_recovery`. We are developing an implementation of PGHI on python.

To generate the magnitudes from TiFGAN-M , please use:
```
cd specgan/generate_commands
python 64md_8k.py
```
The output of that script is a .mat file containing the generated spectrograms and the reconstructed sounds.

## Pre-trained networks
The checkpoints used for the evaluation of the [paper][paper] can be downloaded [here](https://zenodo.org/record/2562819). Please extract the archiv in the folder `saved_results`. To generate magnitudes using those checkpoints, use on of the following commands:
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

## Evaluation

The Inception Score (IS) is computed with [score.py](https://github.com/chrisdonahue/wavegan/blob/master/eval/inception/score.py) from the WaveGAN repository. To also compute the Fréchet Inception Distance (FID), we submitted an extension of that file which is currently being processed as a [pull request](https://github.com/chrisdonahue/wavegan/pull/23). 

## License & co

The content of this repository is released under the terms of the [GNU3 license](LICENCE.txt).
Please cite our [paper] if you use it.

```
@InProceedings{marafioti2019adversarial,
  author    = {Marafioti, Andr{\'e}s and Perraudin, Nathana{\"e}l and Holighaus, Nicki and Majdak, Piotr},
  title     = {Adversarial Generation of Time-Frequency Features with application in audio synthesis},
  booktitle = {Proc. of the 36th ICML},
  year      = {2019},
  editor    = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume    = {97},
  pages     = {4352--4362},
  address   = {Long Beach, California, USA},
  month     = {09--15 Jun},
  publisher = {PMLR},
  url       = {http://proceedings.mlr.press/v97/marafioti19a.html},
}
```
