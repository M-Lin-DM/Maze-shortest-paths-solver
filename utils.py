import numpy as np
from numba import njit


@njit
def rot_m(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def is_row_in_array(row, array):
    return np.any(np.all(array == row, axis=1))


def normalize_field(mu):
    mag = np.sqrt(np.nansum(mu ** 2, axis=0))
    mu = np.divide(mu, mag, out=np.zeros_like(mu), where=np.logical_and(mag != 0, ~np.isnan(mag)))
    return mu
