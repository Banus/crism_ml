"""Plotting utilities."""
import bisect
from textwrap import wrap

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch

import crism_ml.preprocessing as pre
import crism_ml.lab as cl

# whitepoints
_WHITE = {
    'a': (1.0985, 1., 0.3558),      # CIE standard illuminant A
    'c': (0.9807, 1., 1.1822),      # CIE standard illuminant C
    'e': (1., 1., 1.),              # Equal-energy radiator
    # Profile Connection Space (PCS) illuminant
    'icc': (0.964202880859375, 1., 0.82489013671875),
    # DIE standard illuminant D50
    'd50': (0.9641986557609, 1., 0.82511648322104),
    'd55': (0.9568, 1., 0.9214),    # DIE standard illuminant D55
    'd65': (0.95047, 1., 1.08883)   # IE standard illuminant D65
}

_DEFAULT_PRIMARIES = ((0.64, 0.33), (0.3, 0.6), (0.15, 0.06))


def _rgb_xyz_matrix(primaries=_DEFAULT_PRIMARIES, white=_WHITE['d65']):
    m_prime = np.asarray([[x/y for x, y in primaries],
                          [1 for _ in primaries],
                          [(1-x-y)/y for x, y in primaries]])
    return m_prime.dot(np.diag(np.linalg.inv(m_prime).dot(white))).T


def _chromatic_adapter(wfrom, wto, adapt='Bradford'):
    """Convert between different whitepoints."""
    if adapt == 'Bradford':
        _chad = np.asarray([[0.8951, 0.2664, -0.1614],
                            [-0.7502, 1.7135, 0.0367],
                            [0.0389, -0.0685, 1.0296]])
    elif adapt == 'VonKries':
        _chad = np.asarray([[0.40024, 0.70760, -0.08081],
                            [-0.22630, 1.16532, 0.04570],
                            [0.0, 0.0, 0.91822]])
    else:
        _chad = np.eye(3)

    rgbe, rgbs = np.array(wto).dot(_chad.T), np.array(wfrom).dot(_chad.T)
    return np.linalg.solve(_chad, np.diag(rgbe / rgbs)).dot(_chad).T


# adapted from: https://pydoc.net/pwkit/0.8.15/pwkit.colormaps/
#  different from makecform('srgb2lab') and aligned to rgb2lab in Matlab
def _srgb2lab(srgb, source_wp='d50', target_wp='icc', matlab=True):
    """Convert from sRGB to CIE LAB.

    Same behavior as rgb2lab for source_wp='d65' and dest_wp='d65'; close to
    makecform('srgb2lab') with default parameters.
    """
    if matlab:
        if source_wp == 'd65':
            _linsrgb_to_xyz = np.asarray([
                [0.412456439089692, 0.212672851405623, 0.019333895582329],
                [0.357576077643909, 0.715152155287818, 0.119192025881303],
                [0.180437483266399, 0.072174993306560, 0.950304078536368]])
        elif source_wp == 'd50':
            # sRGB.icm matrix; slightly different than the d50 from primaries
            _linsrgb_to_xyz = np.asarray([
                [0.436065673828125, 0.222488403320313, 0.013916015625],
                [0.385147094726563, 0.716873168945313, 0.097076416015625],
                [0.143066406250000, 0.060607910156250, 0.714096069335938]])
        else:
            raise ValueError(f'{source_wp} source whitepoint not supported.')
    else:
        _linsrgb_to_xyz = _rgb_xyz_matrix(white=source_wp)

    # sRGB to linear-rgb to XYZ
    gamma = ((srgb + 0.055) / 1.055)**2.4
    scale = srgb / 12.92
    linsrgb = np.where(srgb > 0.04045, gamma, scale)
    xyz = np.dot(linsrgb, _linsrgb_to_xyz)  # to XYZ

    if source_wp != target_wp:
        xyz = xyz.dot(_chromatic_adapter(_WHITE[source_wp], _WHITE[target_wp]))

    # to CIE LAB
    norm = xyz / np.asarray(_WHITE[target_wp])
    scale = 7.787037 * norm + 16.0/116
    mapped = np.where(norm > 0.008856, norm**(1/3), scale)

    cielab = np.empty_like(xyz)
    cielab[..., 0] = 116 * mapped[..., 1] - 16
    cielab[..., 1] = 500 * (mapped[..., 0] - mapped[..., 1])
    cielab[..., 2] = 200 * (mapped[..., 1] - mapped[..., 2])

    return cielab


# adapted from: https://www.mathworks.com/matlabcentral/fileexchange/\
#  29702-generate-maximally-perceptually-distinct-colors
def distinguishable_colors(n_colors, background=(1.0, 1.0, 1.0)):
    """Generate perceptually distinct colors given a background.

    Adapted from Tim Holy's `perceptually distinct colors`_ script in Matlab.

    .. _perceptually distinct colors: https://www.mathworks.com/matlabcentral/\
        fileexchange/29702-generate-maximally-perceptually-distinct-colors

    Parameters
    ----------
    n_colors: int
        size of the palette to generate
    background: tuple, list
        background color as RGB values in [0,1]; it may be a list of colors

    Returns
    -------
    colors: ndarray
        a color palette as an array of size n_colors x 3 array
    """
    rgbs = np.stack(np.meshgrid(*[np.linspace(0, 1, 30) for _ in range(3)]),
                    axis=-1).reshape((-1, 3))
    labs, bglab = _srgb2lab(rgbs), _srgb2lab(np.asarray(background))
    bglab = np.atleast_2d(bglab)

    colors = np.zeros((n_colors, 3))
    mindists = np.min(cdist(labs, bglab), axis=1)
    lastlab = bglab[-1]

    for idx in range(n_colors):
        dists = cdist(labs, [lastlab]).squeeze()
        mindists = np.minimum(dists, mindists)  # min distance to all
        best = np.argmax(mindists)
        colors[idx, :], lastlab = rgbs[best, :], labs[best, :]

    return colors


CLS_COLORS = dict(zip(sorted(cl.CLASS_NAMES), distinguishable_colors(
    len(cl.CLASS_NAMES), background=(0, 0, 0))))
CLS_COLORS[-1] = CLS_COLORS[255] = (0.3, 0.3, 0.3)
CLS_COLORS[0] = (0.6, 0.6, 0.6)


def _imadjust(src, tol=5, vin=(0, 255), vout=(0, 255)):
    """Adjust image histogram."""
    # from: https://stackoverflow.com/a/44611551
    tol = max(0, min(100, tol))
    if tol > 0:
        hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]
        cum = np.cumsum(hist)

        size = src.shape[0] * src.shape[1]
        lb_, ub_ = size * tol / 100, size * (100 - tol) / 100
        vin = (bisect.bisect_left(cum, lb_), bisect.bisect_left(cum, ub_))

    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs_ = src - vin[0]
    vs_[src < vin[0]] = 0
    vd_ = vs_*scale + 0.5 + vout[0]
    vd_[vd_ > vout[1]] = vout[1]

    return vd_


def get_false_colors(pixspec, badpix, channels=(233, 103, 20)):
    """Return the spectra as false color image on selected bands.

    Convert a spectral cube to an RGB image, using a median filter of size 17
    around selected bands for the R, G and B channels. The pixels are
    normalized first with L2 and then with min-max; finally, histogram
    equalization is applied to each channel.

    Parameters
    ----------
    pixspec: ndarray
        spectra shaped as a (h, w, n_channels) array
    badpix: ndarray
        boolean mask of bad pixels to be excluded from equalization
    channels: tuple
        tuple of three indices specifing the R, G and B bands

    Returns
    -------
    img: ndarray
        the false color image with channels normalized between 0 and 1
    """
    shape = badpix.shape
    badpix, pixspec = badpix.ravel(), pixspec.reshape((-1, pixspec.shape[-1]))

    lsize, rsize = 8, 9  # median filter size: 17
    img = np.stack([np.median(pixspec[:, max(i - lsize, 0):i + rsize], axis=1)
                    for i in channels]).T

    goodpx = img[~badpix, :]
    img /= np.mean(np.sqrt(np.einsum('ij,ij->i', goodpx, goodpx)), axis=0,
                   keepdims=True)

    vmin = np.min(img[~badpix, :], keepdims=True)
    vmax = np.max(img[~badpix, :], keepdims=True)
    img = 255*pre.norm_minmax(img, vmin=vmin, vmax=vmax).reshape(shape + (-1,))

    img = np.stack([_imadjust(im) for im in np.rollaxis(img, 2)], axis=2)
    return img / 255.0


def get_overlay(img, coords, color=(1, 0, 0), dilate=True):
    """Add patch overlay on an image with the given color.

    Parameters
    ----------
    img: ndarray
        a false-color image with channels in the [0,1] range
    coords: ndarray
        a npix x 2 array of x,y coordinates of patch points
    color: tuple
        color of the overlay (default: red)
    dilate: bool
        use a 3x3 morphological dilation on the patch

    Returns
    -------
    res: ndarray
        a copy of the image with the patch overlay
    """
    xx_, yy_ = coords.T
    region = np.full(img.shape[:2], False, dtype=bool)
    region[yy_, xx_] = True
    if dilate:
        region = binary_dilation(region)

    res = img.copy()
    res[region, :] = color
    return res


def _patches_to_regions(avgs, im_shape, probs):
    """Merge patches with the same prediction.

    Optionally compute alpha transparency from the prediction confidence.
    """
    regions, alphas = {}, None
    for avg in avgs:
        regions[avg['pred']] = regions.get(avg['pred'], []) + [avg['coords']]
    regions = {k: np.concatenate(v) for k, v in regions.items()}

    if probs is not None:
        alphas = {}
        for kls, coor in regions.items():
            xx_, yy_ = coor.T
            alphas[kls] = probs[np.ravel_multi_index((yy_, xx_), im_shape)]

    return regions, alphas


def _plot_class_regions(img, regions, alpha, colors):
    """Plot class regions using optional alpha.

    Plot class regions from a dictionary of flattened matrix indices as
    overlay over 'img'; 'alpha' must have the same number of elements as
    'regions' and indicate the alpha value of the pixel.
    """
    res = img.copy()
    for kls, reg in regions.items():
        xx_, yy_ = reg.T

        if alpha is None:
            res[yy_, xx_, :] = colors[kls]
        else:
            k_alpha = alpha[kls][:, None]
            res[yy_, xx_, :] = (1 - k_alpha) * res[yy_, xx_, :] + \
                k_alpha * colors[kls]
    return res


def _plot_class_predictions(img, pred, probs, colors):
    """Plot the predictions.

    Using optionally their confidence (probs) as alpha.
    """
    res = img.copy()
    for kls in np.unique(pred):
        if kls == 0:
            continue
        mask = (pred == kls).reshape(img.shape[:2])
        if probs is None:
            res[mask, :] = colors[kls]
        else:
            res[mask, :] = (1 - probs[pred == kls, None]) * res[mask, :] + \
                probs[pred == kls, None] * colors[kls]
    return res


def show_classes(img, pred, probs=None, **kwargs):
    """Plot classes with an optional legend.

    The function accepts both an array of per-pixel labels, or a list of
    patches detecteed by 'crism_ml.train.evaluate_regions' and plot the
    predictions as overlay on the image.

    Parameters
    ----------
    img: ndarray
        the image to overlay
    pred: ndarray, list
        either an array of class predictions or a list of regions with
        coordinates
    probs: ndarray
        prediction confidence, used as alpha for the overlay
    n_kls_max: int
        maximum number of classes to visualize (default: 15, all if <=0)
    colors: dict
        custom per-class colors; defaults to CLS_COLORS
    with_legend: bool
        adds a legend on the left side of the image (default: True)
    save_to: str
        specifies a filename to save the plot
    crop_to: tuple
        image range in the format returned by crism_ml.preprocessing.crop_image
        to remove invalid pixels on the border
    """
    # pylint: disable=too-many-locals
    n_kls_max = kwargs.pop('n_max', 15)  # max number of displayed classes
    colors = kwargs.pop('colors', CLS_COLORS)
    with_legend = kwargs.pop('with_legend', True)
    save_to = kwargs.pop('save_to', None)
    crop_to = kwargs.pop('crop_to', None)
    if kwargs:
        raise ValueError(f"Unrecognized parameters: {list(kwargs.keys())}")

    if isinstance(pred, list):  # pred is a list of patches
        regions, alpha = _patches_to_regions(pred, img.shape[:2], probs=probs)
        classes = [k for k, _ in sorted(
            ((r, len(v)) for r, v in regions.items()), key=lambda x: -x[1])]

        if n_kls_max > 0:
            classes = classes[:n_kls_max]    # keep largest n_max classes
            regions = {k: r for k, r in regions.items() if k in classes}
        res = _plot_class_regions(img, regions, alpha, colors)
    else:
        classes = [k for k, _ in sorted(
            zip(*np.unique(pred, return_counts=True)), key=lambda x: -x[1])]
        classes = [k for k in classes if k]  # remove null class

        if n_kls_max > 0:
            classes = classes[:n_kls_max]    # keep largest n_max classes
            pred[~np.isin(pred, classes)] = 0
        res = _plot_class_predictions(img, pred, probs, colors)

    if crop_to is not None:
        yy_, xx_ = (slice(*c) for c in crop_to)
        res = res[yy_, xx_, :]

    plt.figure()
    plt.imshow(res, interpolation='none')
    plt.axis("off")

    if with_legend:
        names = cl.CLASS_NAMES if any(c in cl.ALIASES_EVAL for c in classes) \
            else cl.BROAD_NAMES
        legend = [Patch(color=colors[k], label=names[k]) for k in classes]
        plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc='upper left',
                   borderaxespad=0.)

    if save_to is not None:
        plt.savefig(save_to, dpi=200, bbox_inches='tight')


def plot_spectra(spectrum, lab_spectrum, unratioed, attr, overlay=None):
    """Plot the detected and reference spectra with min-max normalization.

    Parameters
    ----------
    spectrum: ndarray
        average patch spectrum (in red, may be None)
    lab_spectrum: ndarray
        the corresponding laboratory spectrum (in blue, may be None)
    unratioed: ndarray
        average unratioed spectrum on the patch (in green, may be None)
    attr: dict
        attributes of the plot: 'title' (required) and 'id' (optional, 5-digit
        image id).
    overlay: ndarray
        image with the patch overlay; if specified, it is plotted on the left
    """
    if overlay is None:
        _, ax2 = plt.subplots()
    else:
        plt.figure()  # pre-existing axes will be ignored
        (ax1, ax2) = (plt.subplot(g) for g in
                      gridspec.GridSpec(1, 2, width_ratios=(1, 1.5)))

        text = f"Image ID: {attr['id']}" if 'id' in attr else ""
        if lab_spectrum is not None:
            lrange = np.max(lab_spectrum) - np.min(lab_spectrum)
            text += f"\nRange: {lrange:.5f}"

        ax1.imshow(overlay)
        ax1.axis('off')
        if text:
            ax1.text(0.05, -0.3, text,
                     transform=ax1.transAxes, verticalalignment='bottom',
                     bbox=dict(facecolor='none', edgecolor='black'))

    spectrum = pre.norm_minmax(spectrum)
    xx_ = pre.BANDS[:len(spectrum)]

    ax2.plot(xx_, spectrum, color='red')
    if lab_spectrum is not None:
        lab_spectrum = pre.norm_minmax(lab_spectrum)
        ax2.plot(xx_, lab_spectrum, color='blue')
    if unratioed is not None:
        unratioed = pre.norm_minmax(unratioed)
        ax2.plot(xx_, unratioed, color='green')

    ax2.set(xlim=[xx_[0], xx_[-1]], ylim=[0, 1.01], xlabel=r"$\mu$m",
            xticks=np.arange(1, 2.8, 0.2), yticks=np.arange(0, 1.1, 0.1),
            title='\n'.join(wrap(attr['title'], 35)), aspect='equal')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.grid('on')
    plt.tight_layout()


def plot_spectrum_density(ax_, data, n_max=500, show_mean=True, color='b'):
    """Plot multiple blended spectra to show the per-band distribution.

    Plots all the spectra using an alpha value scaled to have a roughly
    constant opacity regardless of the number of spectra; if their number
    exceeds n_max, the spectra are subsampled.

    Parameters
    ----------
    ax_: matplotlib.axes
        plot area
    data: ndarray
        spectra to plot as a npix x n_chan array
    n_max: int
        maximum number of spectra to plot (default: 500)
    show_mean: bool
        overlay the mean of the spectra with no transparency
    color: str, tuple
        color of the plots (default: blue)
    """
    if show_mean:
        ax_.plot(np.mean(data, axis=0), color=color)
    if len(data) > n_max:
        step = len(data) // int(n_max) + 1
        data = data[::step, :]
    ax_.plot(data.T, color=color, alpha=5/len(data))


if __name__ == '__main__':
    pass
