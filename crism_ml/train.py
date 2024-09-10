"""Training and evaluation of the GMM model on hyperspectral images."""
import argparse
import logging
import os
import pickle  # nosec
import time

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import crism_ml.preprocessing as cp
import crism_ml.plot as cpl
import crism_ml.lab as cl
from crism_ml.io import cache_to, loadmat, load_image, image_shape
from crism_ml.models import HBM, HBMPrior
from crism_ml import N_JOBS, CONF

# these classes have a default model weight vector associated with them
WEIGHT_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 40, 36,
                  37, 38, 39]  # 35


def iteration_weights(labels=None):
    """Return probability weights per class and model.

    Parameters
    ----------
    labels: list
        restrict the weights to the given class labels

    Returns
    -------
    ww_: ndarray
        array 15 x N classe with per-model and per-class weights
    """
    def _a(lst):  # for broadcasting
        return np.array([lst]).T

    ww_ = np.zeros((15, len(WEIGHT_CLASSES)))   # n iter, n classes
    # H2O ice, CO2 ice, gypsum
    ww_[np.r_[3:7, 8:12], :3] = _a([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # hydroxyl sulfate
    ww_[3:11, 3] = [0.2, 0.1, 0.05, 0.05, 0.1, 0.2, 0.2, 0.1]
    # hematite
    ww_[3:14, 4] = [0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05]
    # nontronite, saponite
    ww_[np.r_[3:7, 9:14], 5:7] = _a([
        .2, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05])
    # prehnite, jarosite, serpentine
    ww_[np.r_[3:7, 8:12], 7] = [0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1, 0.1]
    ww_[3:12, 8] = [0.2, 0.05, 0.2, 0.1, 0.05, 0.05, 0.15, 0.15, 0.05]
    ww_[3:13, 9] = [0.1, 0.1, 0.05, 0.05, 0.4, 0.05, 0.1, 0.05, 0.05, 0.05]
    ww_[3:10, 10] = [0.1, 0.2, 0.2, 0.1, 0.15, 0.15, 0.1]  # alunite
    # akaganeite, fe/ca co3, beidellite, kaolinite, bassanite
    ww_[np.r_[3:7, 10:14], 11] = [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.4, 0.05]
    ww_[np.r_[3:7, 8:12], 12] = [0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1, 0.1]
    ww_[3:12, 13:15] = _a([0.2, 0.2, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05])
    ww_[np.r_[3:7, 10:13], 15] = [0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1]
    # epidote
    ww_[np.r_[3:7, 9:13, 14], 16] = [
        0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.2]
    # al smectite, mg sulfate, mg cl salt, illite, analcime, kieserite
    ww_[3:12, 17] = [0.2, 0.2, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05]
    ww_[np.r_[3:7, 10:13], 18:20] = _a([0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1])
    ww_[np.r_[3:7, 8:12], 20] = [0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1, 0.1]
    ww_[np.r_[3:7, 10:13], 21] = [0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1]
    ww_[3:12, 22] = [0.2, 0.05, 0.1, 0.1, 0.2, 0.05, 0.05, 0.05, 0.2]
    # hydrated silica, copiapite (hydrated sulfate)
    ww_[np.r_[3:7, 8:13], 23:25] = _a([
        0.2, 0.05, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05, 0.05])
    # CO3, chlorite
    ww_[np.r_[3, 5:7, 9:14], 25] = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ww_[np.r_[3:7, 8:12], 26] = [0.2, 0.1, 0.05, 0.05, 0.1, 0.2, 0.2, 0.1]
    # flat categories
    ww_[3:14, 27:34] = _a([
        0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05])
    ww_[3, 34] = 1
    ww_[0:4, :] = ww_[3:4, :] / 4

    if labels is not None:
        missing = set(labels) - set(WEIGHT_CLASSES)
        if missing:
            missing = ', '.join(str(k) for k in missing)
            raise ValueError(f"Missing weights for classes: {missing}.")
        ww_ = ww_[:, [WEIGHT_CLASSES.index(c)
                      for c in labels if c in WEIGHT_CLASSES]]

    return ww_


def feat_masks(as_intervals=False):
    """Feature masks per iteration for normal and bland pixels.

    Parameters
    ----------
    as_intervals: bool
        return the interval starting and ending points instead of the indices

    Returns
    -------
    maskb: ndarray
        per-model band indices for the bland models
    masks: ndarray
        per-model band indices for the mineral models
    """
    def _to_list(mask):
        return [np.concatenate([np.arange(*t) for t in sl]) for sl in mask]

    masks = [[(3, 92, 4), (109, 142, 4), (159, 248, 4)],
             [(4, 93, 4), (110, 143, 4), (160, 245, 4)],
             [(5, 94, 4), (111, 144, 4), (161, 246, 4)],
             [(6, 95, 4), (112, 145, 4), (162, 247, 4)],
             [(44, 75)], [(104, 135)], [(114, 145)], [(154, 185)],
             [(164, 195)], [(174, 205)], [(184, 215)], [(194, 225)],
             [(204, 235)], [(214, 245)], [(59, 90)]]
    maskb = [[(4, 95), (109, 145), (159, 248)]]

    if as_intervals:
        return maskb, masks
    return _to_list(maskb), _to_list(masks)


def default_data_loader(datadir):
    """Load the default mineral dataset.

    Loads the ``CRISM_labeled_pixels_ratioed`` dataset; it is invoked by
    'load_data'.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixlabs: ndarray
        spectra labels
    pixims: ndarray
        ids of the images
    """
    data = loadmat(os.path.join(datadir, "CRISM_labeled_pixels_ratioed.mat"))
    pixspec = data['pixspec'][:, :cp.N_BANDS]
    pixims = data['pixims'].squeeze()
    pixlabs = cl.relabel(data['pixlabs'].squeeze(), cl.ALIASES_TRAIN)

    return pixspec, pixlabs, pixims


@cache_to("dataset.npz", use_version=True)
def load_data(datadir):
    """Load the mineral dataset and removes spikes.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixlabs: ndarray
        spectra labels
    pixims: ndarray
        ids of the images
    """
    loader = CONF.get('data_loader', None)
    loader = default_data_loader if loader is None else loader

    pixspec, pixlabs, pixims = loader(datadir)
    logging.info("Loaded ratioed dataset.")

    logging.info("Removing spikes...")
    pixspec = cp.remove_spikes(
        pixspec, CONF['despike_params']).astype(np.float32)
    logging.info("Done.")

    return pixspec, pixlabs, pixims


def default_unratioed_loader(datadir):
    """Load the default bland dataset.

    Loads the ``CRISM_bland_unratioed`` dataset; it is invoked by
    'load_data_unratioed'.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixims: ndarray
        ids of the images
    """
    data = loadmat(os.path.join(datadir, "CRISM_bland_unratioed.mat"))
    return data['pixspec'], data['pixims'].squeeze()


@cache_to("dataset_bland.npz", use_version=True)
def load_unratioed_data(datadir):
    """Load bland pixels to train blandess detectors.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixlabs: ndarray
        spectra labels (all set to 0)
    pixims: ndarray
        ids of the images
    """
    loader = CONF.get('bland_data_loader', None)
    loader = default_unratioed_loader if loader is None else loader

    pixspec, pixims = loader(datadir)
    logging.info("Loaded unratioed dataset.")

    bad = np.sum(cp.spikes(pixspec[:, :cp.N_BANDS], 27, 10, mask=True),
                 axis=1) > 0

    pixspec, pixims = pixspec[~bad, :], pixims[~bad]
    return pixspec, np.zeros_like(pixims, dtype=np.int16), pixims


@cache_to("bmodel.pkl", use_version=True)
def train_model_bland(datadir, fin):
    """Load dataset and train bland pixel model.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored
    fin: list
        list of feature masks (one per model)

    Returns
    -------
    bmodels: list[HBM]
        list of trained models
    """
    xtrain, ytrain, ids = load_unratioed_data(datadir)
    prior = HBMPrior(**CONF['bland_model_params'])

    def _train(fm_):
        gmm2 = HBM(prior=prior)
        gmm2.fit(cp.normr(xtrain[:, fm_]), ytrain, ids)
        return gmm2

    ts_ = time.time()
    if len(fin) == 1:
        bmodels = [_train(fin[0])]
    else:
        bmodels = Parallel(n_jobs=N_JOBS)(delayed(_train)(fm_) for fm_ in fin)
    logging.info("Training bland models took %.3f seconds", time.time() - ts_)

    return bmodels


@cache_to("model.pkl", use_version=True)
def train_model(datadir, fin):
    """Load dataset and train mineral model.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored
    fin: list
        list of feature masks (one per model)

    Returns
    -------
    models: list[HBM]
        list of trained models
    """
    xtrain, ytrain, ids = load_data(datadir)

    def _train(fm_):
        gmm2 = HBM(only_class=True, prior=HBMPrior(**CONF['model_params']))
        gmm2.fit(cp.norm_minmax(xtrain[:, fm_], axis=1), ytrain, ids)
        return gmm2

    ts_ = time.time()
    if len(fin) == 1:
        models = [_train(fin[0])]
    else:
        models = Parallel(n_jobs=N_JOBS)(delayed(_train)(fm_) for fm_ in fin)
    logging.info("Training models took %.3f seconds", time.time() - ts_)

    return models


def compute_bland_scores(if_, bmodels):
    """Compute bland scores, with parallelism if using an ensemble.

    Parameters
    ----------
    if_: ndarray
        array n_spectra x n_channels of unratioed spectra
    bmodels: tuple
        tuple (models, fin) of list of models and list of their feature masks

    Returns
    -------
    slog: ndarray
        blandness scores for the spectra
    """
    models, fin = bmodels
    if len(models) == 1:  # single model, return it
        return models[0].predict_proba(cp.normr(if_[:, fin[0]]))[:, 0]

    # return ensemble average
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(lambda x, m: m.predict_proba(x)[:, 0])(
            cp.normr(if_[:, f]), m) for m, f in zip(*bmodels))
    return np.add.reduce(scores)


def compute_scores(if_, models, ww_):
    """Compute model scores, with parallelism if using an ensemble.

    Parameters
    ----------
    if_: ndarray
        array n_spectra x n_channels of ratioed spectra
    bmodels: tuple
        tuple (models, fin) of list of models and list of their feature masks

    Returns
    -------
    sumlog: ndarray
        array n_spectra x n_classes with classification scores for the spectra
    """
    mods, fin = models
    if len(mods) == 1:  # single model, return it
        return ww_[0] * mods[0].predict_proba(
            cp.norm_minmax(if_[:, fin[0]], axis=1), llh=False)

    # return ensemble average
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(lambda x, m: m.predict_proba(x, llh=False))(cp.norm_minmax(
            if_[:, f], axis=1), m) for m, f in zip(*models))
    return np.add.reduce([s * w for s, w in zip(scores, ww_)])


def _merge_clays(pred):
    """Merge Al clays together because they are hard to differentiate."""
    smectite, clays = CONF['clays']
    pred[np.isin(pred, clays)] = smectite
    return pred


def filter_predictions(probs, classes, merge_clays=True, thr=0.0, kls_thr=()):
    """Return predicted probabilities, removing low-confidence entries.

    Parameters
    ----------
    probs: ndarray
        array n_pixel x n_classes of prediction probabilities
    classes: ndarray
        map from class position in the probability array to class label
    merge_clay: bool
        merges predictions of Al smectites (default: True)
    thr: float
        global class confidence threshold (if kls_thr is empty)
    kls_thr: tuple
        pair of (low, high) thresholds for class confidences; the former for
        well represented classes (five or more dataset images) and the latter
        for the rest

    Returns
    -------
    pred: ndarray
        the filtered predictions, with low-confidence predictions set to 0
    pred0: ndarray
        the predictions before filtering
    pp_: ndarray
        the confidence of the predictions
    """
    pred = np.argmax(probs, axis=-1)
    pp_ = np.take_along_axis(probs, pred[..., None], axis=-1).squeeze()
    pred = _merge_clays(classes[pred]) if merge_clays else classes[pred]
    pred0 = pred.copy()  # to analyse unfiltered predictions

    if kls_thr:
        thr_low, thr_high = kls_thr
        high_classes = [1, 2, 3, 4, 10, 12, 16, 17, 25, 29, 33, 35, 40, 36, 37,
                        38, 39]
        pred[np.isin(pred, high_classes) & (pp_ < thr_high)] = 0
        pred[(~np.isin(pred, high_classes)) & (pp_ < thr_low)] = 0
    else:
        pred[pp_ < thr] = 0   # below significance level

    return pred, pred0, pp_


def _to_coords(indices, shape):
    """Convert a list of indices into a list of x,y coordinates."""
    coords = np.flip(np.stack(np.unravel_index(indices, shape)), axis=0)
    return coords.T.astype(np.uint16)


def _region_size(classes):
    """Get class threshold; by default, 5 for each class."""
    return np.full((len(classes),), 5)


def evaluate_regions(if_, im_shape, pred, pp_, **kwargs):
    """Compute contiguous regions and refine predictions.

    Parameters
    ----------
    if_: ndarray
        spectra after spike removal and before mineral classification
    im_shape: tuple
        shape of the image as (height, width)
    pred: ndarray
        image predictions as a flattened array
    pp_: ndarray
        prediction confidences as a flattened array
    if0: ndarray
        spectra after bad pixel removal but before spike removal
    dilate: int
        dilation to apply before finding the connected components
    region_size: function
        function returning the region size thresholds for the given classes

    Returns
    -------
    avgs: list
        a list of detected patches with the following fields:

        pred: int
            predicted class
        avg: ndarray
            average ratioed spectrum on the patch
        size: int
            size of the patch
        coords: ndarray
            coordinates of the pixels in the patch
        coords_full: ndarray
            coordinates of the pixels, including the ones with confidences
            lower than CONF['match_threshold']
    """
    if0 = kwargs.pop('if0', None)   # unratioed spectra
    dilate = kwargs.pop('dilate', 2)  # dilation factor for connected comp.
    region_size = kwargs.pop('region_size', _region_size)
    if kwargs:
        raise ValueError(f"Unrecognized parameters: {list(kwargs.keys())}")

    avgs = []
    for kls, thr, regs in cp.regions(pred.reshape(im_shape), region_size,
                                     dilate=dilate):
        for reg in regs:
            reg_good = np.full_like(pred, False, dtype=bool)
            reg_good[reg] = True
            reg_good &= pp_ > CONF['match_threshold']

            if np.sum(reg_good) < thr:
                continue
            unique_cols = np.unique(
                np.unravel_index(reg_good.nonzero(), im_shape)[1])
            if CONF['multi_column'] and len(unique_cols) < 2:
                continue

            avgs.append({'pred': kls, 'avg': np.mean(if_[reg_good, :], axis=0),
                         'size': np.sum(reg_good),
                         'coords': _to_coords(reg_good.nonzero(), im_shape),
                         'coords_full': _to_coords(reg, im_shape)})
            if if0 is not None:
                avgs[-1].update({'avg0': np.mean(if0[reg_good, :], axis=0)})

    return avgs


def _merge_region(regs, kls):
    """Merge region attributes."""
    def _l(lst, field):
        return [x[field] for x in lst]

    sizes = _l(regs, 'size')
    avg = sum(s*x for s, x in zip(sizes, _l(regs, 'avg'))) / np.sum(sizes)

    res = {'pred': kls, 'coords': np.concatenate(_l(regs, 'coords')),
           'coords_full': np.concatenate(_l(regs, 'coords_full')),
           'size': np.sum(sizes), 'avg': avg}
    if 'avg0' in regs[0]:
        res.update(avg0=sum(
            s*x for s, x in zip(sizes, _l(regs, 'avg0'))) / np.sum(sizes))

    return res


def merge_regions(avgs, merge_classes=True):
    """Merge regions with the same label.

    Parameters
    ----------
    avgs: list[dict]
        list of regions from 'evaluate_regions'
    merge_classes: bool
        merge classes with the same BROAD_NAMES (default: True)

    Returns
    -------
    regions: list[dict]
        merged regions with the same format as the input regions
    """
    regions = {}
    for avg in avgs:
        pred = avg['pred']
        # use alias if defined, otherwise keep class unchanged
        pred = cl.ALIASES_EVAL.get(pred, pred) if merge_classes else pred
        regions[pred] = regions.get(pred, []) + [avg]

    return [_merge_region(regs, kls) for kls, regs in regions.items()]


def run_on_images(images, datadir, workdir, thresholds=(0.5, 0.7), plot=False):
    """Train models and run them on a set of images.

    Parameters
    ----------
    images: list[str]
        list of filenames of images to process
    datadir: str
        path where the datasets are stored
    workdir: str
        working directory (it will be created if it doesn't exists)
    thresholds: tuple
        thresholds to use to select predictions (see 'filter_predictions')
    plot: bool
        if True, saves per-image and per-region plots in 'workdir/plots'
    """
    # pylint: disable=too-many-locals
    os.makedirs(workdir, exist_ok=True)

    fin0, fin = feat_masks()
    bmodels = train_model_bland(datadir, fin0)
    models = train_model(datadir, fin)
    ww_ = iteration_weights(models[0].classes)

    for im_path in images:
        im_, _ = os.path.splitext(os.path.basename(im_path))
        logging.info("Processing: %s", im_)
        mat = load_image(im_path)

        ts_ = time.time()
        if_, rem = cp.filter_bad_pixels(mat['IF'])
        logging.info("Removing bad pixels took %.3f seconds",
                     time.time() - ts_)

        ts_ = time.time()
        im_shape = image_shape(mat)
        if1 = cp.remove_spikes_column(
            if_.reshape(*im_shape, -1), 3, 5).reshape(if_.shape)
        logging.info("Removing column spikes took %.3f seconds",
                     time.time() - ts_)

        ts_ = time.time()
        slog = compute_bland_scores(if1, (bmodels, fin0))
        logging.info("Bland scores took %.3f seconds", time.time() - ts_)

        ts_ = time.time()
        slog_inf = cp.replace(slog, rem, -np.inf).reshape(im_shape)
        if2 = cp.ratio(if1.reshape(*im_shape, -1), slog_inf).reshape(if_.shape)
        logging.info("Ratioing took %.3f seconds", time.time() - ts_)

        ts_ = time.time()
        ifm = cp.remove_spikes(if2.copy(), CONF['despike_params'])
        logging.info("Spike removal took %.3f seconds", time.time() - ts_)

        # classify
        ts_ = time.time()
        sumlog = compute_scores(ifm, (models, fin), ww_)
        logging.info("Classification took %.3f seconds", time.time() - ts_)

        pred, pred0, pp_ = filter_predictions(sumlog, models[0].classes,
                                              kls_thr=thresholds)

        ts_ = time.time()
        avgs = evaluate_regions(
            if2, im_shape, cp.replace(pred, rem, 0), pp_, if0=if_)
        regs = merge_regions(avgs)
        logging.info("Region parsing took %.3f seconds", time.time() - ts_)

        if plot:
            plotdir = os.path.join(workdir, "plots")
            os.makedirs(os.path.join(plotdir, im_), exist_ok=True)

            # plot full segmentation map
            rem = rem.reshape(im_shape)
            im_fc = cpl.get_false_colors(if_, rem)
            cpl.show_classes(im_fc/2, regs, crop_to=cp.crop_region(rem),
                             save_to=os.path.join(plotdir, f"{im_}.pdf"))

            # per-patch plots
            for r_id, avg in enumerate(avgs):
                title = f"{cl.BROAD_NAMES[avg['pred']]} - {avg['size']}px"
                attr = {'title': title, 'id': im_}
                cpl.plot_spectra(avg['avg'][:cp.N_BANDS], None,
                                 avg['avg0'][:cp.N_BANDS], attr,
                                 overlay=cpl.get_overlay(im_fc, avg['coords']))
                plt.savefig(
                    os.path.join(plotdir, im_, f"{avg['pred']}-{r_id+1}.png"),
                    dpi=120, bbox_inches='tight')
                plt.close()

        # save per-patch and per-image results
        with open(os.path.join(workdir, f"{im_}.pkl"), 'wb') as fid:
            pickle.dump([avgs, regs], fid, pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(os.path.join(workdir, im_),
                            [pp_, pred, pred0, slog])


def get_parser():
    """Get parser for command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A script to train the HBM models and run the evaluation"
                    "on a list of images")

    parser.add_argument('image', type=str, nargs='+',
                        help="CRISM images to process")
    parser.add_argument('--datapath', '-d', type=str, default="datasets",
                        help="directory where the datasets are stored.")
    parser.add_argument('--workdir', '-w', type=str, default="workdir",
                        help="directory where the results are stored.")
    parser.add_argument('--thr', '-t', type=float, nargs='+',
                        default=(0.5, 0.7),
                        help="confidence thresholds for easy and hard classes")
    parser.add_argument("--plot", action="store_true",
                        help="save detailed per-region plots")
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args()
    run_on_images(args.image, args.datapath, args.workdir, args.thr, args.plot)
