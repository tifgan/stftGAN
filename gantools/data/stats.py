import numpy as np
import functools

def power_law_with_cutoff(x,k=2,c=1e4):
    """Power law with cutoff.
    
    H: x>=1
    Arguments:
    x : numpy array
    """
    res = np.zeros(shape=x.shape)
    mask = x>c
    maski = mask==False
    res[maski] = power_law(x[maski],k=k)
    res[mask] = power_law(c,k=k)*np.exp(-(x[mask]-c)/c)
    return res

def power_law(x,k=2):
    """Power law for x>=1.
    
    p(x) = a x^(-k)
    """
    assert(k>1)
    a = k-1
    return a/(x**k)

def power_law_cdf(x, k=2):
    """CDF for power law for x>=1.
    
    c(x) = 1 - 1/(x^(k-1))
    """
    assert(k>1)
    a = k-1
    return 1 - 1/(x**a)

def power_law_cdf_inv(x, k=2):
    """Inverse CDF for power law.
    
    k=2 for now.
    c(x) = 1/(1-x)
    """
    assert(k==2)
    return 1/(1-x)


def power_law_wcf_cdf(x, c):
    """Power law with cutoff.
    
    H: x>=1
    Arguments:
    x : numpy array
    c : cuttoff
    """
    res = np.zeros(shape=x.shape)
    mask = x>c
    maski = mask==False
    res[maski] = power_law_cdf(x[maski],k=2)
    res[mask] = (c-1.0)/c + 1/c * cutoff(x[mask]/c-1)
    return res

def power_law_wcf_cdf_inv(x, c):
    """Inverse power law with cutoff.
    
    H: x>=1
    Arguments:
    x : numpy array
    c : cuttoff
    """
    res = np.zeros(shape=x.shape)
    mc = power_law_cdf(c,k=2)
    mask = x>mc
    maski = mask==False
    res[maski] = power_law_cdf_inv(x[maski],k=2)
    res[mask] = np.round( c*(cutoff_inv(c*(x[mask]-1)+1) +1) )
    return res


def cutoff(x):
    return 1 - np.exp(-x)

def cutoff_inv(x, order=2):
    return -np.log(1-x)

def laplacian_map_from_cdf_forward(x, cdf):
    cp = cdf(x)
    return -np.log(1-cp)

def laplacian_map_from_cdf_backward(x, cdf, cdf_inv, clip_max=None):
    if clip_max:
        v = np.array([clip_max])
        x_lim = laplacian_map_from_cdf_forward(v, cdf)
        x = np.clip(x,0,x_lim)
    cl = 1 - np.exp(-x)
    return cdf_inv(cl)



def stat_forward_old(x, c=1e4, shift=6):
    cdf = functools.partial(power_law_wcf_cdf, c=c)
    sv = laplacian_map_from_cdf_forward(np.array([1 + shift]), cdf)[0]
    return laplacian_map_from_cdf_forward(x+1 + shift, cdf) - sv

def stat_backward_old(x, c=1e4, shift=6):
    clip_max = c*30
    v = np.array([clip_max])
    x_lim = stat_forward_old(v, c=c, shift=shift)
    x = np.clip(x,0,x_lim[0])
    cdf = functools.partial(power_law_wcf_cdf, c=c)
    cdf_inv = functools.partial(power_law_wcf_cdf_inv, c=c)
    sv = laplacian_map_from_cdf_forward(np.array([1 + shift]), cdf)[0]
    return np.round(laplacian_map_from_cdf_backward(x+ sv, cdf, cdf_inv, clip_max=clip_max)-1-shift)
