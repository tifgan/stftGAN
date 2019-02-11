# stftGAN


**paper**: [Adversarial Generation of Time-Frequency Features
with application in audio synthesis][paper]

[paper]: https://arxiv.org/abs/...........

TODO: Add intro + nice image


## Installation

First clone the repository.

```
git clone https://github.com/tifgan/stftGAN.git 
```

#### Requirements

*We hightly recommend to work in a virtual environnement.*

You can simply install those packages with the following command:
```
pip install -r requirements.txt
```

TODO: Andrès, what about LTFAT

## Datasets

Here are some datasets we used to train TifGAN:

- [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz)
- [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz)
- [Drum sound effects](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz)

The data should be extracted in the  `data` folder. On the notebook inside the folder there are instructions to generate a dataset from audio files.

## Train a TifGAN

TODO: Andrès, add which script is used to train the GAN

## Generate samples

TODO: Andrès, add which script is used to generate sample/reconstruct the phase

## Evaluation

The Inseption Score (IS) and Frechet Inception Distance (FID) can be computed using:

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
