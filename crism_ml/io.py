"""Utilities for input/output operations."""
import logging
import os
from pathlib import Path
import pickle  # nosec
from functools import wraps

import numpy as np
from crism_ml import CONF, USE_CACHE


ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")


def cache_to(cache_fname, use_version=False):
    """Decorate a function to cache its output results to file.

    The function expect a name (not a path) of a file and stores the caches in
    the '.cache' directory in the project root.

    Parameters
    ----------
    cache_fname: str
        name of the cache file
    use_version: bool
        prepends the version of the configuration to avoid cache mixups
        (default: False)
    """
    def _decorator(func):
        if not USE_CACHE:
            return func

        @wraps(func)
        def _cached(*args, **kwargs):
            fname = f"{CONF['version']}_{cache_fname}" if use_version \
                else cache_fname
            fname = os.path.join(CACHE_DIR, fname)
            os.makedirs(CACHE_DIR, exist_ok=True)
            is_npz = fname.endswith('.npz')

            try:
                if is_npz:
                    arr = np.load(fname)
                    res = (arr[k] for k in arr.files)
                else:
                    with open(fname, 'rb') as fid:
                        res = pickle.load(fid)  # nosec
                logging.info("Loading from cache: %s", fname)
                return res
            except IOError:
                res = func(*args, **kwargs)
                if is_npz:
                    np.savez_compressed(fname, *res)
                else:
                    with open(fname, 'wb') as fid:
                        pickle.dump(res, fid, pickle.HIGHEST_PROTOCOL)
                return res
        return _cached
    return _decorator


def loadmat(fname):
    """Load matlab files.

    Load Matlab files using the scipy interfaces and falling back to 'mat73'
    for the new HDF5 format.

    Parameters
    ----------
    fname: str
        Matlab file to open

    Returns
    -------
    mat: dict
        a dictionary storing the Matlab variables
    """
    # pylint: disable=import-outside-toplevel
    try:
        from scipy.io import loadmat as _loadmat
        return _loadmat(fname)
    except NotImplementedError as ex:
        # scipy loads only files with version 7.3 or earlier
        try:
            from mat73 import loadmat as _loadmat73
            return _loadmat73(fname)
        except ImportError as mat_ex:
            raise ex from mat_ex


def image_shape(mat):
    """Get the image shape from the pixel x and y coordinates."""
    return (np.max(mat['y']), np.max(mat['x']))


def _generate_envi_header(lbl_fname):
    """Generate a HDR file from the LBL file when the former is missing."""
    # see: https://github.com/jlaura/crism/blob/master/csas.py
    fbase, _ = os.path.splitext(lbl_fname)

    with open(lbl_fname, 'r') as fid:
        for line in fid:
            if "LINES" in line:
                lines = int(line.split("=")[1])
            if "LINE_SAMPLES" in line:
                samples = int(line.split("=")[1])
            if "BANDS" in line:
                bands = int(line.split("=")[1])

    with open(f"{fbase}.hdr", 'w') as fid:
        fid.write(
            f"ENVI\nsamples = {samples}\nlines   = {lines}\nbands   = {bands}"
            "\nheader offset = 0\nfile type = ENVI Standard\ndata type = 4\n"
            "interleave = bil\nbyte order = 0")


def crism_to_mat(fname, flatten=False):
    """Convert a CRISM ENVI image to a Matlab-like dictionary.

    Loads an ENVI image as a Matlab-like dictionary with spectra (IF) and pixel
    coordinates (x, y). If the header (.hdr) is not found, it is automatically
    generated from a .lbl file (using the approach in the
    `CRISM spectral calculator`_); if neither is available, an error is raised.

    .. _CRISM spectral calculator: https://github.com/jlaura/crism/blob/\
        master/csas.py

    Parameters
    ----------
    fname: str
        ENVI file to open (.hdr or .img)
    flatten: bool
        flatten an image array to (npix, nchan) and saves the coordinates to
        the x,y fields (default: False)

    Returns
    -------
    mat: dict
        a dictionary storing the spectra and the pixels coordinates (if flatten
        is True)
    """
    # pylint: disable=import-outside-toplevel
    from spectral.io import envi, spyfile

    band_select = np.r_[433:185:-1, 170:-1:68]

    fbase, _ = os.path.splitext(fname)
    try:
        img = envi.open(f"{fbase}.hdr")
    except spyfile.FileNotFoundError:
        _generate_envi_header(f"{fbase}.lbl")
        img = envi.open(f"{fbase}.hdr")

    arr = img.load()

    mdict = {'IF': arr[:, :, band_select]}
    if flatten:  # use coordinate arrays for indexing
        xx_, yy_ = np.meshgrid(np.arange(arr.shape[1]),
                               np.arange(arr.shape[0]))
        mdict.update({'x': xx_.ravel() + 1, 'y': yy_.ravel() + 1})
        mdict['IF'] = mdict['IF'].reshape((-1, len(band_select)))

    return mdict


def load_image(fname):
    """Try to load a .mat file and fall back to ENVI if not found."""
    try:
        return loadmat(fname)
    except (FileNotFoundError, NotImplementedError, ValueError):
        return crism_to_mat(fname, flatten=True)


if __name__ == '__main__':
    pass
