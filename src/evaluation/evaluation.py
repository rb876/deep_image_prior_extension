# -*- coding: utf-8 -*-
import numpy as np

def get_median_psnr_history(psnr_histories):
    """
    Parameters
    ----------
    psnr_histories : array-like
        PSNR history or histories.  The last dimension is iterations, all other
        dimensions are reduced using ``np.median``.

    Returns
    -------
    median_psnr_history : :class:`numpy.ndarray`
        Median PSNR history.
    """
    psnr_histories = np.asarray(psnr_histories)
    median_psnr_history = np.median(psnr_histories,
                                    axis=range(psnr_histories.ndim - 1))

    return median_psnr_history

def get_psnr_steady(psnr_histories, start, stop):
    """
    Determine the steady PSNR, defined as the median of the median PSNR history
    in the interval ``start:stop``.

    Parameters
    ----------
    psnr_histories : array-like
        PSNR history or histories.  The last dimension is iterations, all other
        dimensions are reduced using ``np.median``.
    start : int or None
        Start of the interval (`start` argument to :class:`slice`).
    stop : int or None
        End of the interval (`stop` argument to :class:`slice`).

    Returns
    -------
    psnr_steady : float
        Steady PSNR.
    """
    median_psnr_history = get_median_psnr_history(psnr_histories)
    psnr_steady = np.median(median_psnr_history[start:stop])

    return float(psnr_steady)

def get_rise_time_to_baseline(psnr_histories,
                              baseline_psnr_steady, remaining_psnr=0.5):
    """
    Determine the rise time, defined as the number of iterations until the
    difference between the median PSNR history and `baseline_psnr_steady` falls
    below `remaining_psnr` (at least for one iteration).

    Parameters
    ----------
    psnr_histories : array-like
        PSNR history or histories.  The last dimension is iterations, all other
        dimensions are reduced using ``np.median``.
    baseline_psnr_steady : scalar
        Steady PSNR of a baseline.
    remaining_psnr : float, optional
        Value below which the difference between the median PSNR history and
        `baseline_psnr_steady` falls at rise time, thereby defining the latter.
        The default is ``0.5``.

    Returns
    -------
    rise_time_to_baseline : int
        Rise time (number of iterations).
    """
    median_psnr_history = get_median_psnr_history(psnr_histories)
    rise_time = int(np.argwhere(
            median_psnr_history > baseline_psnr_steady - remaining_psnr)[0][0])

    return rise_time
