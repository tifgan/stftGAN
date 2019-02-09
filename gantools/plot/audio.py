import numpy as np
from scipy.io import wavfile
from IPython.core.display import HTML
from matplotlib import pyplot as plt

# this is a wrapper that take a filename and publish an html <audio> tag to listen to it


def wavPlayer(filepath):
    """ will display html 5 player for compatible browser

    Parameters :
    ------------
    filepath : relative filepath with respect to the notebook directory ( where the .ipynb are not cwd)
               of the file to play

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens
    """

    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>

    <body>
    <audio controls="controls" style="width:600px" >
      <source src="files/%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """ % (filepath)
    display(HTML(src))


def play_sound(x, fs, filename=None):
    if filename is None:
        filename = str(np.random.randint(10000)) + '.wav'
    wavfile.write(filename, np.int(fs), (x * (2**15)).astype(np.int16))
    wavPlayer(filename)


def plot_signals(sigs, nx=1, ny=1, *args, **kwargs):
    """
    Draw multiple images. This function conveniently draw multiple images side
    by side.

    Parameters
    ----------
    sigs : List of Signales
        - Matrix [ n , sx ]
    """
    ndim = len(sigs.shape)
    nimg = sigs.shape[0]

    if ndim == 1:
        raise ValueError('The input seems to contain only one signal')
    elif ndim == 2:
        if nx * ny > nimg:
            raise ValueError("Not enough signals")
    else:
        raise ValueError('The input contains to many dimensions')

    f, ax = plt.subplots(ny, nx, sharey=True, figsize=(4 * nx, 3 * ny))
    it = 0
    lim = np.max(np.abs(sigs))
    xlim = (-lim, lim)
    for i in range(nx):
        for j in range(ny):
            if nx == 1 or ny == 1:
                ax[j + i].plot(sigs[it])
                ax[j + i].set_ylim(xlim)
            else:
                ax[j, i].plot(sigs[it])
                ax[j, i].set_ylim(xlim)
            it += 1