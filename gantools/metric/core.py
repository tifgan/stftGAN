"""This module define the base classes for the metrics of GANs."""

import numpy as np
import tensorflow as tf
from gantools.plot.plot_summary import PlotSummaryLog, PlotSummaryPlot


class TFsummaryHelper(object):
    """Helper class for tensorflow summaries."""

    def __init__(self, name, group='', stype=0):
        """Create a statistical object.

        Arguments
        ---------
        * name: name of the statistic (preferably unique)
        * group: group for the statistic
        """
        self._name = name
        self._group = group
        self._stype = stype

    def add_summary(self,  collections=None):
        """Add a tensorflow summary.

        stype: summary type. 
               * 0 scalar
               * 1 image
               * 2 histogram
               * 3 curves
               * 4 simple unique curve
        """

        name = self.group + '/' + self.name
        print("Add summary for "+name)

        if self.stype == 0:
            self._placeholder = tf.placeholder(tf.float32, name=name)
            tf.summary.scalar(name, self._placeholder, collections=[collections])
        elif self.stype == 1:
            self._placeholder = tf.placeholder(
                tf.float32, shape=[None, None], name=name)
            tf.summary.image(name, self._placeholder, collections=[collections])
        elif self.stype == 2:
            self._placeholder = tf.placeholder(tf.float32, shape=[None], name=name)
            tf.summary.histogram(name, self._placeholder, collections=[collections])
        elif self.stype == 3:
            self._placeholder = tf.placeholder(tf.float32, name=name)
            tf.summary.scalar(name, self._placeholder, collections=[collections])
            if self._log:
                self._plot_summary = PlotSummaryLog(
                    self.name, self.group, collections=[collections])
        elif self.stype == 4:
            self._plot_summary = PlotSummaryPlot(
                self.name, self.group, collections=[collections])

        else:
            raise ValueError('Wrong summary type')

    def compute_summary(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class.")

    @property
    def group(self):
        return self._group
    
    @property
    def name(self):
        return self._name

    @property
    def stype(self):
        return self._stype
    

class Statistic(TFsummaryHelper):
    """Base class for a statistic."""

    def __init__(self, func, *args, **kwargs):
        """Create a statistical object.

        Arguments
        ---------
        * func: function to compute the statistic
        """
        super().__init__(*args, **kwargs)
        self._func = func

    def compute_summary(self, dat, feed_dict={}):
        feed_dict[self._placeholder] = self(dat)
        if self.stype == 4:
            feed_dict = self._plot_summary.compute_summary(
                self._saved_real_stat,
                feed_dict=feed_dict)
        return feed_dict

    def __call__(self, *args, **kwargs):
        """Compute the statistic."""
        return self._func(*args, **kwargs)


class Metric(TFsummaryHelper):
    """Base metric class."""

    def __init__(self, name, group='', recompute_real=True, **kwargs):
        """Initialize the statistic.

        Argument
        --------
        * name: name of the statistic (preferably unique)
        * recompute_real: recompute the real statistic (default True)
        """
        super().__init__(name=name, group=group, **kwargs)
        self._preprocessed = False
        self._recompute_real = recompute_real
        self._last_metric = None


    def preprocess(self, real, **kwargs):
        """Do the preprocessing.

        This function is designed to do all the precomputation that can be done with the real data.
        If this computaation can be done only once, i.e. the real data is always the same, then set recompute_real to False
        """
        self._preprocessed = True

    def __call__(self, fake, real=None):
        """Compute the metric."""
        if self._recompute_real or ((not self.preprocessed) and (real is not None)):
            self.preprocess(real)
        elif (not self.preprocessed) and (real is not None):
            raise ValueError("The real data need to be preprocessed first!")
        self._last_metric = self._compute(fake, real)
        return self._last_metric

    def _compute(self, fake, real):
        """Compute the metric."""
        raise NotImplementedError("This is an abstract class.")

    def compute_summary(self, fake, real, feed_dict={}):
        feed_dict[self._placeholder] = self(fake, real)
        return feed_dict

    @property
    def preprocessed(self):
        """Return True if the preprocessing been done."""
        return self._preprocessed

    @property
    def last_metric(self):
        return self._last_metric
    


class StatisticalMetric(Metric):
    """Statistically based metric."""

    def __init__(self, statistic, order=2, log=False, normalize=False, wasserstein=False, **kwargs):
        """Initialize the StatisticalMetric.

        Arguments
        ---------
        * name: name of the metric (preferably unique)
        * statistic: a statistic object
        * order: order of the norm (default 2, Froebenius norm)
        * log: compute the log of the stat before the norm (default False)
        * recompute_real: recompute the real statistic (default True)
        * normalize: normalize the metric (default False)
        * wasserstein: use the wasserstein metric
        """
        name = statistic.name + '_l' + str(order)
        if log:
            name += 'log'
        super().__init__(name, statistic.group, **kwargs)
        self._order = order
        self._log = log
        self.statistic = statistic
        self._saved_stat = None
        self._normalize = normalize
        self._wasserstein = wasserstein

    def preprocess(self, real, rerun=True):
        """Compute the statistic on the real data."""
        if rerun or (not self.preprocessed):
            super().preprocess(real)
            print('Compute real statistics: '+self.group+'/'+self.name)
            self._saved_real_stat = self.statistic(real)

    def _compute_stats(self, fake, real=None):
        self._saved_fake_stat = self.statistic(fake)

    def _compute(self, fake, real):
        # The real is not vatiable is not used as the stat over real is
        # computed only once
        # print("Compute summaries: "+self.group+'/'+self.name)
        self._compute_stats(fake, real)
        fake_stat = self._saved_fake_stat
        real_stat = self._saved_real_stat
        if isinstance(real_stat, tuple):
            rs = real_stat[0]
            fs = fake_stat[0]
        else:
            rs = real_stat
            fs = fake_stat
        if self._log:
            rs = 10*np.log10(rs + 1e-2)
            fs = 10*np.log10(fs + 1e-2)
        if self._wasserstein:
            self._last_metric = wasserstein_distance(rs, fs, normalize=self._normalize, w=self._saved_real_stat[1])
        else:
            self._last_metric = np.mean(np.abs(rs - fs)**self._order)
            if self._normalize:
                self._last_metric /= np.mean(np.abs(rs)**self._order)
        return self._last_metric

    def compute_summary(self, fake, real, feed_dict={}):
        super().compute_summary(fake, real, feed_dict)
        if self.stype == 3:
            feed_dict = self._plot_summary.compute_summary(
                self._saved_real_stat[1],
                self._saved_real_stat[0],
                self._saved_fake_stat[0],
                feed_dict=feed_dict)
        return feed_dict

    @property
    def real_stat(self):
        if self._saved_real_stat:
            return self._saved_real_stat
        else:
            raise ValueError("The statistic has not been computed yet")

    @property
    def fake_stat(self):
        if self._saved_fake_stat:
            return self._saved_fake_stat
        else:
            raise ValueError("The statistic has not been computed yet")


class StatisticalMetricLim(StatisticalMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lim = None

    def preprocess(self, real, **kwargs):
        """Compute the statistic on the real data."""
        super().preprocess(real, **kwargs)

        self._lim = self._saved_real_stat[2]

    def _compute_stats(self, fake, real=None):
        self._saved_fake_stat = self.statistic(fake, lim=self._lim)


class MetricSum(Metric):
    """This class represent a sum of metrics."""
    def __init__(self, metrics, recompute_real=True, **kwargs):
        """Initialize the StatisticalMetric.

        Arguments
        ---------
        * name: name of the metric (preferably unique)
        * metrics: list of metrics object
        * group: group for the metric
        """
        super().__init__(recompute_real=recompute_real, **kwargs)
        self._metrics = metrics

        for metric in self._metrics:
            metric._recompute_real = recompute_real

    def preprocess(self, real, **kwargs):
        for metric in self._metrics:
            metric.preprocess(real, **kwargs)
        super().preprocess(real, **kwargs)

    def _compute(self, fake, real=None):
        score = 0
        for metric in self._metrics:
            score += metric(fake, real)
        return score

    def add_summary(self,  *args, **kwargs):
        for metric in self._metrics:
            metric.add_summary(*args, **kwargs)
        super().add_summary(*args, **kwargs)

    def compute_summary(self, fake, real, feed_dict={}):
        score = 0
        for metric in self._metrics:
            feed_dict = metric.compute_summary(fake, real, feed_dict)
            score += metric.last_metric
        feed_dict[self._placeholder] = score
        return feed_dict


def wasserstein_distance(x, y, w=None, safe=True, normalize=True):
    """Wasserstein distance for 1D vectors."""
    if w is None:
        w = np.arange(x.shape[0])
    weights = np.diff(w)
    if normalize:
        x = x/np.sum(x)
        y = y/np.sum(y)
    if safe:
        assert (x.shape == y.shape == w.shape)
        np.testing.assert_almost_equal(np.sum(x), np.sum(y))
        assert ((x >= 0).all())
        assert ((y >= 0).all())
        assert ((weights >= 0).all())
    cx = np.cumsum(x)[:-1]
    cy = np.cumsum(y)[:-1]
    return np.sum(weights * np.abs(cx - cy)) / (w[-1] - w[0])
