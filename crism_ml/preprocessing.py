"""Preprocessing utility functions."""
import logging

import numpy as np

from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from scipy.ndimage import label, binary_dilation
from joblib import Parallel, delayed

from crism_ml import N_JOBS

N_BANDS = 248

BANDS = np.array([
    1.021, 1.02755, 1.0341, 1.04065, 1.0472, 1.05375, 1.0603, 1.06685,
    1.07341, 1.07996, 1.08651, 1.09307, 1.09962, 1.10617, 1.11273, 1.11928,
    1.12584, 1.13239, 1.13895, 1.14551, 1.15206, 1.15862, 1.16518, 1.17173,
    1.17829, 1.18485, 1.19141, 1.19797, 1.20453, 1.21109, 1.21765, 1.22421,
    1.23077, 1.23733, 1.24389, 1.25045, 1.25701, 1.26357, 1.27014, 1.2767,
    1.28326, 1.28983, 1.29639, 1.30295, 1.30952, 1.31608, 1.32265, 1.32921,
    1.33578, 1.34234, 1.34891, 1.35548, 1.36205, 1.36861, 1.37518, 1.38175,
    1.38832, 1.39489, 1.40145, 1.40802, 1.41459, 1.42116, 1.42773, 1.43431,
    1.44088, 1.44745, 1.45402, 1.46059, 1.46716, 1.47374, 1.48031, 1.48688,
    1.49346, 1.50003, 1.50661, 1.51318, 1.51976, 1.52633, 1.53291, 1.53948,
    1.54606, 1.55264, 1.55921, 1.56579, 1.57237, 1.57895, 1.58552, 1.5921,
    1.59868, 1.60526, 1.61184, 1.61842, 1.625, 1.63158, 1.63816, 1.64474,
    1.65133, 1.65791, 1.66449, 1.67107, 1.67766, 1.68424, 1.69082, 1.69741,
    1.70399, 1.71058, 1.71716, 1.72375, 1.73033, 1.73692, 1.74351, 1.75009,
    1.75668, 1.76327, 1.76985, 1.77644, 1.78303, 1.78962, 1.79621, 1.8028,
    1.80939, 1.81598, 1.82257, 1.82916, 1.83575, 1.84234, 1.84893, 1.85552,
    1.86212, 1.86871, 1.8753, 1.8819, 1.88849, 1.89508, 1.90168, 1.90827,
    1.91487, 1.92146, 1.92806, 1.93465, 1.94125, 1.94785, 1.95444, 1.96104,
    1.96764, 1.97424, 1.98084, 1.98743, 1.99403, 2.00063, 2.00723, 2.01383,
    2.02043, 2.02703, 2.03363, 2.04024, 2.04684, 2.05344, 2.06004, 2.06664,
    2.07325, 2.07985, 2.08645, 2.09306, 2.09966, 2.10627, 2.11287, 2.11948,
    2.12608, 2.13269, 2.1393, 2.1459, 2.15251, 2.15912, 2.16572, 2.17233,
    2.17894, 2.18555, 2.19216, 2.19877, 2.20538, 2.21199, 2.2186, 2.22521,
    2.23182, 2.23843, 2.24504, 2.25165, 2.25827, 2.26488, 2.27149, 2.2781,
    2.28472, 2.29133, 2.29795, 2.30456, 2.31118, 2.31779, 2.32441, 2.33102,
    2.33764, 2.34426, 2.35087, 2.35749, 2.36411, 2.37072, 2.37734, 2.38396,
    2.39058, 2.3972, 2.40382, 2.41044, 2.41706, 2.42368, 2.4303, 2.43692,
    2.44354, 2.45017, 2.45679, 2.46341, 2.47003, 2.47666, 2.48328, 2.4899,
    2.49653, 2.50312, 2.50972, 2.51632, 2.52292, 2.52951, 2.53611, 2.54271,
    2.54931, 2.55591, 2.56251, 2.56911, 2.57571, 2.58231, 2.58891, 2.59551,
    2.60212, 2.60872, 2.61532, 2.62192, 2.62853, 2.63513, 2.64174, 2.64834,
    2.80697, 2.81358, 2.8202, 2.82681, 2.83343, 2.84004, 2.84666, 2.85328,
    2.85989, 2.86651, 2.87313, 2.87975, 2.88636, 2.89298, 2.8996, 2.90622,
    2.91284, 2.91946, 2.92608, 2.9327, 2.93932, 2.94595, 2.95257, 2.95919,
    2.96581, 2.97244, 2.97906, 2.98568, 2.99231, 2.99893, 3.00556, 3.01218,
    3.01881, 3.02544, 3.03206, 3.03869, 3.04532, 3.05195, 3.05857, 3.0652,
    3.07183, 3.07846, 3.08509, 3.09172, 3.09835, 3.10498, 3.11161, 3.11825,
    3.12488, 3.13151, 3.13814, 3.14478, 3.15141, 3.15804, 3.16468, 3.17131,
    3.17795, 3.18458, 3.19122, 3.19785, 3.20449, 3.21113, 3.21776, 3.2244,
    3.23104, 3.23768, 3.24432, 3.25096, 3.2576, 3.26424, 3.27088, 3.27752,
    3.28416, 3.2908, 3.29744, 3.30408, 3.31073, 3.31737, 3.32401, 3.33066,
    3.3373, 3.34395, 3.35059, 3.35724, 3.36388, 3.37053, 3.37717, 3.38382,
    3.39047, 3.39712, 3.40376, 3.41041, 3.41706, 3.42371, 3.43036, 3.43701,
    3.44366, 3.45031, 3.45696, 3.46361, 3.47026, 3.47692
])


def _resample_convhull(sig, bands):
    """Resample the signal using the convex hull."""
    ext_bands = np.concatenate([[bands[0]], bands, [bands[-1]]])
    ext_sig = np.concatenate([[0], sig, [0]])

    # the behavior is the same as Matlab's convhull with 'Simplify' set to 1
    conv = ConvexHull(np.stack([ext_bands, ext_sig], axis=1)).vertices
    conv = np.sort(conv)[1:-1] - 1  # shift indices
    conv = conv[conv < N_BANDS]

    if not sig[0]:  # align to Matlab without simplify when leading zeros
        conv = np.concatenate([[0, np.nonzero(sig)[0][0] - 1], conv])

    return interp1d(bands[conv], sig[conv], bounds_error=False,
                    copy=False, assume_sorted=True)(bands)


def remove_continuum(sig, bands=None):
    """Remove the slowly-varying component from the spectrum.

    Ported and simplified from Iordache's Matlab implementation [1]_.

    Parameters
    ----------
    sig: ndarray
        signal to remove the continuum from
    bands: ndarray
        signal bands; defaults to the 248 main bands

    Returns
    -------
    flat_sig: ndarray
        signal without the continuum
    curve: ndarray
        the continuum curve

    References
    ----------
    .. [1] Iordache, Marian-Daniel. (2016). Matlab Code and Demo for Continuum
      Removal. 10.13140/RG.2.1.2885.9285.
    """
    if bands is None:
        bands = BANDS[:N_BANDS]

    sig = np.atleast_2d(sig)
    not_const = np.ptp(sig, axis=1) > 0
    flat_sig = np.zeros(sig.shape)

    curve = [_resample_convhull(s, bands) for s in sig[not_const]]
    with np.errstate(invalid='ignore'):  # nan is fine
        flat_sig[not_const] = sig[not_const] / curve

    return flat_sig, curve


def filter_bad_pixels(pixspec, copy=False):
    """Remove large, infinite or NaN values from the spectra.

    Parameters
    ----------
    pixspec: ndarray
        the set of spectra to clean
    copy: bool
        if a new array must be returned; by default the orginal is overwritten

    Returns
    -------
    pixspec: ndarray
        the cleaned spectra with bad pixels set to the mean of all channels
    rem: ndarray
        boolean mask of bad pixels, with the first n-1 dimensions of pixspec
    """
    if copy:
        pixspec = pixspec.copy()

    bad = (pixspec > 1e3) | ~np.isfinite(pixspec)
    if np.any(bad):
        pixspec[bad] = np.mean(pixspec[~bad])

    rem = np.sum(bad[..., :N_BANDS], axis=-1) > 0
    logging.info("%d bad pixels", np.sum(rem))

    return pixspec, rem.reshape(pixspec.shape[:-1])


def crop_region(rem):
    """Convert bad pixels to a crop rectangle.

    If bad pixels are not shaped like a rectangular border, a warning is
    issued.

    Parameters
    ----------
    rem: ndarray
        boolean mask of bad pixels, it must be shaped like an image.

    Returns
    -------
    crop: tuple
        a tuple ((ymin, ymax), (xmin, xmax)) giving the crop rectangle.
    """
    yy_, xx_ = np.nonzero(~rem)

    mask = np.full(rem.shape, True, dtype=bool)
    mask[yy_[0]:yy_[-1]+1, xx_[0]:xx_[-1]+1] = False
    if np.any(rem != mask):
        logging.warning("Cropping image but invalid pixels inside the box.")

    return ((yy_[0], yy_[-1]+1), (xx_[0], xx_[-1]+1))


def normr(pixspec):
    """L2 normalizarion on the last axis.

    Parameters
    ----------
    pixspec: ndarray
        the set of spectra to normalize

    Returns
    -------
    res: ndarray
        the normalized spectra
    """
    norms = np.sqrt(np.einsum('ij,ij->i', pixspec, pixspec))
    with np.errstate(invalid='ignore'):  # invalid values will be overwritten
        res = pixspec / norms[:, np.newaxis]

    nfeat = pixspec.shape[1]
    # use Matlab's convention of using a normalized 1 for null rows
    res[norms == 0.0, :] = np.ones((nfeat,)) / np.sqrt(nfeat)
    return res


def norm_minmax(pixspec, vmin=None, vmax=None, axis=0):
    """Normalize features in the [0,1] range.

    Parameters
    ----------
    pixspec: ndarray
        the set of spectra to normalize
    vmin: float
        custom minimum value; if None, computed from data
    vmax: float
        custom maximum value; if None, computed from data
    axis: int
        dimension to normalize (default: 0)

    Returns
    -------
    res: ndarray
        the normalized spectra
    """
    if vmin is None or vmax is None:
        vmin = np.min(pixspec, axis=axis, keepdims=True)
        vmax = np.max(pixspec, axis=axis, keepdims=True)

    diff = vmax - vmin
    diff[diff == 0] = 1.0   # avoid division by 0
    return (pixspec - vmin) / diff


def _medfilt1_np(array, size):
    """Numpy implementation, ~ size times slower than the Matlab version."""
    lsize = size // 2
    rsize = size - lsize
    return np.stack([np.median(array[..., max(i-lsize, 0):i+rsize], axis=-1)
                     for i in range(array.shape[-1])], axis=-1)


def _medfilt1_bk(array, size):
    """Implement with bottleneck using NaN padding to have symmetric window."""
    import bottleneck as bk  # pylint: disable=import-outside-toplevel

    rsize, ndim = size - size // 2 - 1, len(array.shape)
    pad = ((0, 0),) * (ndim - 1) + ((0, rsize),)
    array = np.pad(array.astype(np.float64), pad_width=pad,
                   mode='constant', constant_values=np.nan)
    return bk.move_median(array, size, min_count=1, axis=-1)[..., rsize:]


def medfilt1(array, size):
    """Implement the moving median filter with the 'truncate' strategy.

    Re-implementation of Matlab's ``medfilt1`` function on the last dimension
    and using the 'truncate' strategy. It tries to use the fast implementation
    from the'bottleneck' library, with a fallback to Numpy if it is not found.

    Parameters
    ----------
    array: ndarray
        array to filter
    size: int
        the size of the filter

    Returns
    -------
    res: ndarray
        the filtered array
    """
    try:
        return _medfilt1_bk(array, size)
    except ImportError:
        return _medfilt1_np(array, size)


def spikes(pixspec, size, sigma, mask=False):
    """Remove spikes bigger than sigma deviations.

    Replace samples farther than sigma std deviations with the value from a
    moving median filter with window 'size'; if 'mask' is True, it simply
    returns the spike mask.

    Parameters
    ----------
    pixspec: ndarray
        the spectra to filter
    size: int
        the size of the moving median filter
    sigma: float
        max spike size in standard deviations over all the spectra
    mask: bool
        return the bad channels mask if true; replaces them with the median
        otherwise (default)

    Returns
    -------
    ind: ndarray
        boolean mask of the bad channels, per pixels; only if mask=True
    """
    def _mean(arr):
        """Mean with broadcast."""
        return np.mean(arr, keepdims=True)

    pixmed = medfilt1(pixspec, size)
    diff = np.abs(pixmed - pixspec)
    # Matlab uses the unbiased std, passing ddof=1 to align
    ind = diff > _mean(np.mean(diff[..., :N_BANDS], axis=-1)) + \
        sigma * _mean(np.std(diff[..., :N_BANDS], ddof=1, axis=-1))

    if not mask:
        pixspec[ind] = pixmed[ind]
    return ind


def remove_spikes(pixspec, params=((11, 5), (7, 5), (3, 5))):
    """Remove large spikes from spectra with decreasing window sizes.

    Replaces points that exceed by a multiple of the standard deviation of the
    signal with the median over a window, to remove outliers [1]_.

    .. rubric:: References

    .. [1] H. Liu, S. Shah, and W. Jiang. "On-line outlier detection and data
      cleaning." Computers & chemical engineering, 28(9):1635-1647, 2004

    Parameters
    ----------
    pixspec: ndarray
        the spectra to filter
    params: tuple
        list of (size, sigma) parameters for the 'spike' function, to be
        applied in sequence

    Returns
    -------
    pixspec: ndarray
        the spectra without spikes (in-place processing)
    """
    for size, sigma in params:
        spikes(pixspec, size, sigma)

    return pixspec


def remove_spikes_column(pixspec, size, sigma, copy=True):
    """Remove spikes using per-column statistics.

    Parameters
    ----------
    pixspec: ndarray
        spectra to filter with shape height x width x n_channels
    size: int
        size of the median filter
    sigma: float
        max spike size in standard deviations over the column spectra
    copy: bool
        return a copy of the spectra (default: True)

    Returns
    -------
    pixspec: ndarray
        the filtered spectra
    """
    if copy:
        pixspec = pixspec.copy()

    col_avg = np.mean(pixspec[..., :N_BANDS], axis=0)
    diff = np.abs(col_avg - medfilt1(col_avg, size))
    ind = diff > np.mean(diff, axis=-1, keepdims=True) + \
        sigma * np.std(diff, ddof=1, axis=-1, keepdims=True)

    pixmed = medfilt1(pixspec[..., :N_BANDS], size)
    xx_, zz_ = ind.nonzero()
    pixspec[:, xx_, zz_] = pixmed[:, xx_, zz_]

    return pixspec


#
# Ratioing
#

def _y_slice(idx, nrows, win):
    if nrows < 2 * win:
        raise ValueError(f"Not enough rows: {nrows} found, {2 * win} required")

    if idx < win:
        slices = np.r_[0:idx-1, idx+2:2*win-1]
    elif idx >= nrows - win:
        slices = np.r_[nrows-2*win:idx-1, idx+2:nrows-1]
    else:
        slices = np.r_[idx-win+1:idx-1, idx+2:idx+win]
    return slices.astype(np.int32)


def _ratio_win(pixspec, bscore, idx, win, size):
    """Sort best candidates in window and ratio spectra."""
    nrows, ncols = pixspec.shape[:2]
    slices = _y_slice(idx, nrows, win)
    if len(slices) < size:  # not enough candidates
        return np.zeros(pixspec.shape[1:])

    bwin = bscore[slices, :]
    sidx = np.argpartition(-bwin, size, axis=0)[:size]  # highest 'size'

    bwin = np.take_along_axis(bwin, sidx, axis=0)
    # reject if bad pixels or not enough good bland pixels
    is_bad = np.any(np.isinf(bwin), axis=0) | np.isinf(bscore[idx, :])
    # selecting on indices is more efficient than selecting the array
    xx_ = np.repeat(np.arange(ncols, dtype=np.int32), size)
    sidx = np.take_along_axis(slices[:, None], sidx, axis=0).T
    bland = np.mean(pixspec[sidx.ravel(), xx_].reshape((ncols, size, -1)),
                    axis=1)

    normed = pixspec[idx, :, :] / bland
    normed[is_bad | (np.sum(bland, axis=1) == 0), :] = 0

    return normed


def replace(arr, mask, val):
    """Replace values in mask with val; returns a copy of the array.

    Parameters
    ----------
    arr: ndarray
        array of values to replace
    mask: ndarray
        boolean mask of values to replace
    val: float
        value replacing the elements on the mask

    Returns
    -------
    res: ndarray
        a copy of the array with the values replaced
    """
    res = arr.copy()
    res[mask] = val
    return res


def ratio(pixspec, bscore, window=50, size=3):
    r"""Ratioes the spectra with respect to the most bland pixels.

    Ratioes the spectra w.r.t. the most bland pixels in the given window
    radius; if multiple windows are passed, it averages the results.
    Blandness for invalid pixels must be :math:`-\infty`.

    Parameters
    ----------
    pixspec: ndarray
        array of size height x width x n_channels of spectra to ratio
    bscore: ndarray
        array of size height x width of blandness scores (higher is blander)
    window: int, tuple
        size of the search window for bland pixels on the column; if it is a
        tuple, it returns the average of the ratios using each size; the
        default is 50
    size: int
        number of candidate bland pixels to use (default: 3)

    Returns
    -------
    ratioed: ndarray
        the ratioed spectra
    """
    if isinstance(window, (list, tuple)):
        ratios = Parallel(n_jobs=N_JOBS)(delayed(ratio)(
            pixspec, bscore, w, size) for w in window)
        return np.mean(ratios, axis=0)

    return np.stack([_ratio_win(pixspec, bscore, i, window, size)
                     for i in range(pixspec.shape[0])])


def ratio_masked(pixspec, coords, bscore, window=50, size=3):
    """Ratioes selected pixels and also return denominator coordinates.

    Analogue to 'ratio' but restricted to the given coordinates; gives a speed-
    up when the number of spectra is significantly lower than the number of
    pixels in the image.

    Parameters
    ----------
    pixspec: ndarray
        array of size height x width x n_channels of spectra to ratio
    coords: ndarray
        array of size n_points x 2 with the coordinates of pixels to ratio
    bscore: ndarray
        array of size height x width of blandness scores (higher is blander)
    window: int, tuple
        size of the search window for land pixels on the column
    size: int
        number of candidate bland pixels to use (default: 3)

    Returns
    -------
    ratioed: ndarray
        the ratioed spectra
    dens: ndarray
        the denominators used to ratio the pixels
    """
    # pylint: disable=too-many-locals
    if isinstance(window, (list, tuple)):
        # copy kwargs because each call pops the parameters
        ratios, indices = zip(*[ratio_masked(
            pixspec, bscore, coords, w, size) for w in window])
        return np.mean(ratios, axis=0), np.concatenate(indices, axis=1)

    ratioed, out_coords = np.zeros((len(coords), pixspec.shape[-1])), []

    for idx in np.unique(coords[:, 0]):   # unique y coordinate
        slices = _y_slice(idx, pixspec.shape[0], window)
        mask = coords[:, 0] == idx   # select only pixels in mask sharing y
        xx_ = coords[mask, 1]

        bwin = bscore[np.ix_(slices, xx_)]  # cartesian product slice
        sidx = np.argpartition(-bwin, size, axis=0)[:size]
        bwin = np.take_along_axis(bwin, sidx, axis=0)
        is_bad = np.any(np.isinf(bwin), axis=0) | np.isinf(bscore[idx, xx_])

        # select bland pixels
        y_b = np.take_along_axis(slices[:, None], sidx, axis=0).ravel()
        x_b = np.tile(xx_, size)
        bland = np.mean(
            pixspec[y_b, x_b].reshape((size, len(xx_), -1)), axis=0)
        out_coords.append(np.stack([y_b, x_b], axis=1).reshape((-1, size, 2)))

        normed = pixspec[idx, xx_, :] / bland
        normed[is_bad | (np.sum(bland, axis=1) == 0), :] = 0
        ratioed[mask] = normed

    return ratioed, np.concatenate(out_coords, axis=0)


def ratio_colmed(pixspec, rem, midonly=False):
    """Use the median on of column (ColMed) for ratioing ('medianratio').

    .. _Mars wiki: https://marssi.univ-lyon1.fr/wiki/CRISM

    Parameters
    ----------
    pixspec: ndarray
        array of size height x width x n_channels of spectra to ratio
    rem: ndarray
        boolean mask of bad pixels with size height x width
    midonly: bool
        restricts the median to the half interval in the middle (see
        `Mars wiki`_)

    Returns
    -------
    ratioed: ndarray
        the ratioed spectra
    """
    def medcol(idx):
        colwin = pixspec[:, idx, :][~rem[:, idx]]
        nrows = len(colwin)
        colwin = colwin[nrows//4:-nrows//4] if midonly else colwin

        normed = pixspec[:, idx, :] / np.median(colwin, axis=0)
        normed[rem[:, idx]] = 0
        return normed

    return np.stack([medcol(i) for i in range(pixspec.shape[1])], axis=1)


def ratio_colmed_masked(pixspec, coords, rem, midonly=False):
    """Medianratio only on selected pixel coordinates.

    Invalid pixels will be ratioed if they are in 'coords' but they will not be
    used to compute the median.

    Parameters
    ----------
    pixspec: ndarray
        array of size height x width x n_channels of spectra to ratio
    coords: ndarray
        array of size n_points x 2 with the coordinates of pixels to ratio
    rem: ndarray
        boolean mask of bad pixels with size height x width
    midonly: bool
        restricts the median to the half interval in the middle

    Returns
    -------
    ratioed: ndarray
        the ratioed spectra
    dens: ndarray
        the denominators used to ratio the pixels
    """
    ratioed = np.zeros((len(coords), pixspec.shape[-1]))
    dens = np.zeros_like(ratioed)  # ratio denominators

    for idx in np.unique(coords[:, 1]):   # unique x coordinate
        colwin = pixspec[:, idx, :][~rem[:, idx]]
        colwin = colwin[len(colwin)//4:-len(colwin)//4] if midonly else colwin

        mask = coords[:, 1] == idx
        yy_, xx_ = zip(*coords[mask])
        den = np.median(colwin, axis=0)
        ratioed[mask], dens[mask] = pixspec[yy_, xx_] / den, den

    return ratioed, dens


#
# postprocessing
#

def label_to_index(labels, n_obj):
    """Fast label to index conversion when n labels > sqrt(n pixels)."""
    # from: https://stackoverflow.com/a/26888164
    spm = csr_matrix((np.arange(labels.size),
                      (labels.flatten(), np.arange(labels.size))),
                     shape=(n_obj + 1, labels.size))
    return np.split(spm.data, spm.indptr)[1:-1]  # first/last empty


def _get_regions(mask, thr, dilate=0):
    """Get connected components for the given prediction mask."""
    if dilate:
        dil_mask = binary_dilation(mask, structure=np.ones((dilate, dilate)))
        labels, n_obj = label(dil_mask, structure=np.ones((3, 3)))
        labels[~mask] = 0  # dilation is only to get connected components
    else:
        labels, n_obj = label(mask, structure=np.ones((3, 3)))

    return [g for g in label_to_index(labels, n_obj)[1:] if g.size >= thr]


def regions(preds, region_size, dilate=2):
    """Return a list of connected regions from the predictions.

    Given a 2D prediction array, for each class it finds connected components.
    It returns the class, the size threshold and the linear indices of the
    pixels in the region. The 'region_size' parameter is a function that
    returns the size thresholds for the given list of classes.

    Parameters
    ----------
    preds: ndarray
        2D array of class predictions
    region_size: function
        given a list of classes, it returns region size thresholds in the same
        order
    dilate: int
        morphological dilation before computing the connected component; only
        the non-dilated pixels are returned

    Returns
    -------
    regs: list
        list of tuples with class, size threshold and region indices
    """
    classes = np.unique(preds)[1:]  # remove null predictions
    return [(kls, thr, _get_regions(preds == kls, thr, dilate)) for kls, thr in
            zip(classes, region_size(classes))]


if __name__ == '__main__':
    pass
