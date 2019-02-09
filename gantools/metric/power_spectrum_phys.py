#Code based on Andres code from November 2017

import numpy as np
from six.moves import xrange

def part2dens3d(part_pos, box_l, bin_x=128):
    """
    Calculate 3D matter density using numpy histograms
    :param part_pos: particle positions in the shape (N, D), where N is particle number and D is dimension
    :param box_l: box length in comoving Mpc/h
    :param bin_x: desired bins per axis for the histogram
    :return: density field
    """
    hist, _edges = np.histogramdd(np.vstack((part_pos[:, 0], part_pos[:, 1], part_pos[:, 2])).T,
                                  bins=bin_x, range=[[0, box_l], [0, box_l], [0, box_l]])
    del _edges
    return hist


def part2dens2d(part_pos, box_l, bin_x=128):
    """
    Calculate 2D matter density using numpy histograms
    :param part_pos: particle positions in the shape (N, D), where N is particle number and D is dimension
    :param box_l: box length in comoving Mpc/h
    :param bin_x: desired bins per axis for the histogram
    :return: density field
    """
    hist, _edgex, _edgey = np.histogram2d(part_pos[:, 0], part_pos[:, 1], bins=bin_x, range=[[0, box_l], [0, box_l], [0, box_l]])
    del _edgex, _edgey
    return hist


def dens2overdens(density, mean_density=None):
    """
    Calculate the overdensity corresponding to a density field
    :param density: input density field
    :param mean_density: if defined normalisation is calculated according to (density - mean(density)) / mean_density
    :return: overdensity field
    """
    #assert np.ndim(density) == 3, 'density is not 3D'

    if mean_density:
        delta = (density - np.mean(density)) / mean_density
    else:
        mean_density = np.mean(density)
        if mean_density == 0.:
            delta = np.zeros(shape=density.shape)
        else:
            delta = density / mean_density - 1.
    return delta


def power_spectrum(field_x, box_l, bin_k, field_y=None):
    """
        Measures the mass power spectrum of a 3D input field for a given number of bins in Fourier space.
        :param field_x: 3D input field to compute the power spectrum of (typically the overdensity field), dimensionless
        :param box_l: box length of image/cube/box or whatever, units of Mpc or Mpc/h
        :param bin_k: number of bins in Fourier space
        :return: power_k, k: 1D mass power spectrum of field_x, same units as [box_l]**3 and corresponding k values
        """
    # assert np.ndim(field_x) == 3, 'field_x is not 3D'
    box_pix = np.size(field_x, axis=0)  # pixel number per axis
    box_dim = np.ndim(field_x)  # dimension

    # This first 'paragraph' is to create masks of indices corresponding to one Fourier bin each
    _freq = np.fft.fftfreq(n=box_pix, d=box_l / box_pix) * 2 * np.pi
    _rfreq = np.fft.rfftfreq(n=box_pix, d=box_l / box_pix) * 2 * np.pi

    if box_dim == 2:
        _kx, _ky = np.meshgrid(_freq, _rfreq, indexing='ij')
        _k_abs = np.sqrt(_kx ** 2. + _ky ** 2.)
    elif box_dim == 3:
        _kx, _ky, _kz = np.meshgrid(_freq, _freq, _rfreq, indexing='ij')
        _k_abs = np.sqrt(_kx ** 2. + _ky ** 2. + _kz ** 2.)
    else:
        raise ValueError('field_x is not 2D or 3D')
    # The following complicated line is actually only creating a 1D array spanning k-space logarithmically from minimum _k_abs to maximum.
    # To start slightly below the minimum and finish slightly above the maximum I use ceil and floor.
    # To ceil and floor not to the next integer but to the next 15th digit, I multiply by 1e15 before flooring and divide afterwards.
    # Since the ceiled/floored value is actually the exponent used for the logspace, going to the next integer would be way too much.
    _k_log = np.logspace(np.floor(np.log10(np.min(_k_abs[1:])) * 1.e15) / 1.e15,
                         np.ceil(np.log10(np.max(_k_abs[1:])) * 1.e15) / 1.e15, bin_k)

    X = np.fft.rfftn(np.fft.fftshift(field_x)) * (box_l / box_pix) ** box_dim
    if field_y is not None:
        Y = np.conj(np.fft.rfftn(np.fft.fftshift(field_y))) * (box_l / box_pix) ** box_dim

    power_k = np.empty(np.size(_k_log) - 1)
    for i in xrange(np.size(_k_log) - 1):
        mask = (_k_abs >= _k_log[i]) & (_k_abs < _k_log[i + 1])
        if field_y is None:
            if np.sum(mask):
                power_k[i] = np.mean(np.abs(X[mask] ** 2)) / box_l ** box_dim
            else:
                power_k[i] = np.nan
        else:
            if np.sum(mask):
                power_k[i] = np.mean(np.real(X[mask] * np.conj(Y[mask]))) / box_l ** box_dim
            else:
                power_k[i] = np.nan            

    k = (_k_log[1:] + _k_log[:-1]) / 2

    return power_k, k