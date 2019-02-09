import socket
import os

# def data_path(spix=256):  
#     ''' Will be removed in the futur '''
#     return '/scratch/snx3000/nperraud/nati-gpu/data/size{}_splits1000_n500x3/'.format(spix)

def root_path():
    ''' Defining the different root path using the host name '''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        rootpath = '/scratch/snx3000/nperraud/pre_processed_data/' 
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/cosmology/pre_processed_data/'         
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/nbody/preprocessed_data/'
    return rootpath


def celeba_path():
    '''Return the root path of the CelebA dataset.'''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        rootpath = '/scratch/snx3000/nperraud/celeba/'
    elif 'omenx' in hostname:
        rootpath = '/store/nati/dataset/downsampled-celeba/'
    else:
        raise NotImplementedError()
    return rootpath

def medical_path():
    '''Return the root path of the electron microscopy dataset.'''
    hostname = socket.gethostname()
    # Check if we are on pizdaint
    if 'nid' in hostname:
        rootpath = '/scratch/snx3000/nperraud/pre_processed_medical_data/'
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/pre_processed_medical_data/'
    else:
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/'
    return rootpath


def nsynth_path():
    hostname = socket.gethostname()
    if ('nid' in hostname) or ('daint' in hostname):
        rootpath = '/scratch/snx3000/nperraud/data/'
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/Nsynth-gan/'
    else:
        utils_module_path = os.path.dirname(__file__)

        rootpath = utils_module_path + '/../../data/'
    return rootpath


def piano_path():
    hostname = socket.gethostname()
    if ('nid' in hostname) or ('daint' in hostname):
        rootpath = '/scratch/snx3000/nperraud/data/'
    elif 'omenx' in hostname:
        rootpath = '/store/nati/datasets/piano/'
    else:
        utils_module_path = os.path.dirname(__file__)

        rootpath = utils_module_path + '/../../data/'
    return rootpath


def berlin_path():
    hostname = socket.gethostname()
    if 'omenx' in hostname:
        rootpath = '/store/nati/datasets/maps/berlin/'
    else:
        # This should be done in a different way
        utils_module_path = os.path.dirname(__file__)
        rootpath = utils_module_path + '/../../data/maps/berlin/'
    return rootpath