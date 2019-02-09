import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from gantools.plot import audio
# Inspired by Andres


class PlotSummary(object):
    def __init__(self, name, cat, collections=None):
        self._name = name
        self._cat = cat
        self._collections = collections
        self._plot_str = tf.placeholder(tf.string)
        self._build_graph()

    def _build_graph(self):
        '''
        Build the tf graph for creating plot summary
        '''
        image = tf.image.decode_png(self._plot_str, channels=4)
        image = tf.expand_dims(image, 0)

        self._summary = tf.summary.image(
            self._cat + '/' + self._name, image, collections=self._collections)

    def get_placeholder(self):
        return self._plot_str

    def compute_summary(self, *args, feed_dict={}, **kwargs):
        self.plot(*args, **kwargs)
        feed_dict[self._plot_str] = self._get_plot_str()
        return feed_dict

    def produceSummaryToWrite(self, session, *args, **kwargs):
        feed_dict = self.compute_summary( *args, **kwargs)
        return session.run(self._summary, feed_dict=feed_dict)

    def plot(self):
        plt.Figure()
        # ax = plt.gca()
        # N = 10
        # x = np.linspace(0, N - 1, N)
        # y = np.random.rand(N)
        # ax.plot(x, y, label="Random data", color='r')
        # ax.title.set_text(self._name + "\n")
        # ax.title.set_fontsize(11)
        # ax.tick_params(axis='both', which='major', labelsize=10)
        # ax.legend()

    def _get_plot_str(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()


class PlotSummaryLog(PlotSummary):
    def plot(self, x, real, fake):
        super().plot()
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        linestyle = {
            "linewidth": 1,
            "markeredgewidth": 0,
            "markersize": 3,
            "marker": "o",
            "linestyle": "-"
        }
        ax.plot(x, real, label="Real", color='r', **linestyle)
        ax.plot(x, fake, label="Fake", color='b', **linestyle)

        # ax.set_ylim(bottom=0.1)
        ax.title.set_text(self._name + "\n")
        ax.title.set_fontsize(11)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend()

class PlotSummaryPlot(PlotSummary):
    def __init__(self, nx, ny, *args, **kwargs):
        self.nx = nx
        self.ny = ny
        super().__init__(*args, **kwargs)

    def plot(self, x):
        nel = x.shape[0]
        if nel < self.nx*self.ny:
            self.nx = int(np.round(np.sqrt(nel)))
            self.ny = nel // self.nx
        audio.plot_signals(x, nx=self.nx, ny=self.ny)
        # f, ax = plt.subplots(
        #     self.nx, self.ny, sharey=True, figsize=(4 * self.nx, 3 * self.ny))
        # it = 0
        # lim = np.max(np.abs(x))
        # xlim = (-lim, lim)
        # for i in range(self.nx):
        #     for j in range(self.ny):

        #         ax[i, j].plot(x[it, :])
        #         ax[i, j].set_ylim(xlim)
        #         it += 1


