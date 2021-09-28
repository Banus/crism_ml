"""Probabilistic models for classification."""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, softmax  # pylint: disable=no-name-in-module
from scipy.linalg import solve_triangular, solve, eigh, eig


class Model:
    """Generic model interface."""

    def __init__(self):
        """Inititialize classes as empty."""
        self.classes = None

    def fit(self, _data, labels):
        """Train the model on data and labels.

        Parameters
        ----------
        _data: ndarray
            training data
        labels: ndarray
            training labels as a integer array (not one-hot encoding); the
            labels don't have to be contiguous
        """
        self.classes = np.array(sorted(list(set(labels))))

    def _predict_proba(self, data):
        """Return log-likelihood; private template method."""

    def predict_proba(self, data, llh=True):
        """Predicts per-class probabilities.

        Parameters
        ----------
        data: ndarray
            the input data for which the predictions will be computed
        llh: bool
            return the log-likelihood (default), otherwise return normalized
            probabilities
        """
        return softmax(self._predict_proba(data), axis=-1) if not llh else \
            self._predict_proba(data)

    def predict(self, data):
        """Predict classes."""
        return self.classes[np.argmax(self.predict_proba(data), axis=-1)]


def _pca(mat, perc):
    """Return the transformation capturing 'perc' variance of the sample."""
    eiv, eiw = eigh(mat, check_finite=False)

    nd_ = next(iter(np.nonzero(np.cumsum(eiv[::-1] / np.sum(eiv)) > perc)[0]),
               None)
    if nd_ is None:
        return -eiw
    return -eiw[:, -nd_ - 2:]


def _lda(mat, perc):
    """Return optimal separation hyperplane given the covariance."""
    eiv, eiw = eig(mat, check_finite=False)
    eiv, eiw = np.real(eiv), np.real(eiw)

    nd_ = next(iter(np.nonzero(np.cumsum(eiv / np.sum(eiv)) > perc)[0]), None)
    if nd_ is None:
        return eiw
    return eiw[:, :nd_ + 1]


def _get_balanced_stats(data, labels, ids, classes=None):
    """Get per-class mean / std averaged per image."""
    if classes is None:
        classes = np.unique(labels)

    dim, n_k = data.shape[1], len(classes)
    mu_0, s_0 = np.zeros((n_k, dim)), np.zeros((n_k, dim, dim))

    for i, kls in enumerate(classes):
        data_k = data[labels == kls, :]
        id_k = ids[labels == kls].squeeze()
        uid_k = np.unique(id_k)

        for id_ in uid_k:
            data_i = data_k[id_k == id_, :]
            mu_0[i, :] += np.mean(data_i, axis=0)
            s_0[i, :, :] += np.cov(data_i.T)
        s_0[i, :, :] /= len(uid_k)
        mu_0[i, :] /= len(uid_k)

    return mu_0, s_0


#
# 2-level Gaussian Mixture Model
#

@dataclass
class HBMPriorData:
    """Prior parameters for the 2-layer GMM model."""

    mu_0: np.ndarray
    s_0: np.ndarray
    scale: float
    k_0: float
    k_1: float
    wdof: float


class HBMPrior:  # pylint: disable=too-few-public-methods
    """Generates a prior based on the given parameters."""

    def __init__(self, mode=None, perc=0.99):
        """Initialize the prior parameters.

        Parameters
        ----------
        mode: None|'pca'|'lda'
            optional dimensionality reduction using PCA or LDA
        perc: float
            variance captured by dimensionality reduction (only if mode is not
            None)
        """
        self.mode, self.perc, self.mean_cov = mode, perc, None

    def _get_default_prior(self, data, labels, ids, classes):
        """Compute prior parameters from data statistics."""
        dim, n_k = data.shape[1], len(classes)
        mu_0, s_0 = _get_balanced_stats(data, labels, ids, classes)

        if self.mode == 'lda':
            self.mean_cov = np.cov(mu_0.T)
        mu_0, s_0 = np.mean(mu_0, axis=0), np.mean(s_0, axis=0)

        _ones = np.ones((n_k,))
        return HBMPriorData(mu_0=mu_0, scale=1 * _ones, k_0=1 * _ones,
                            s_0=s_0, k_1=100 * _ones, wdof=(dim + 2) * _ones)

    def get_prior(self, data, labels, ids, classes):
        """Return the prior and the (optional) data transformation.

        Parameters
        ----------
        data: ndarray
            the training data to compute the prior values
        labels: ndarray
            the training labels to compute per-class statistics
        ids: ndarray
            image ids to compute per-image statistics
        classes: ndarray
            list of classes for which the prior will be computed

        Returns
        -------
        prior: HBMPriorData
            dataclass storing the prior parameters
        vtr: ndarray
            matrix encoding the dimensionality reduction transformation
        """
        prior, vtr = self._get_default_prior(data, labels, ids, classes), None

        if self.mode == 'pca':
            vtr = _pca(prior.s_0, self.perc)
        elif self.mode == 'lda':
            vtr = _lda(solve(prior.s_0, self.mean_cov, assume_a='pos'),
                       self.perc)

        if vtr is not None:
            prior.s_0, prior.mu_0 = vtr.T @ prior.s_0 @ vtr, prior.mu_0 @ vtr

        return prior, vtr


class HBM(Model):  # pylint: disable=too-many-instance-attributes
    r"""Hierarchical Bayesian Model (HBM).

    The mineral distribution is modeled by a Gaussian distribution with a local
    (per image) prior, in turn derived from a global prior. The distribution is
    given by:

    .. math::

        \text{Data model:} & \quad \boldsymbol{x}_{ijk} \sim \mathcal{N}(
            \boldsymbol{\mu}_{jk}, \Sigma_k) \\
        \text{Local prior:} & \quad \boldsymbol{\mu}_{jk} \sim \mathcal{N}(
            \boldsymbol{\mu}_k, \Sigma_{k}\kappa_1^{-1}) \\
        \text{Global prior:} & \quad \boldsymbol{\mu}_k \sim \mathcal{N}(
            \boldsymbol{\mu}_0,\Sigma_k\kappa_0^{-1}) \quad \Sigma_{k} \sim
            \mathcal{IW} (\Sigma_0,m)

    where *k*, *j* and *i* indicate the class, instance (image) and pixel
    respectively; :math:`\mathcal{IW}` is the `Inverse Wishart`_ distribution.
    We compute the *posterior predictive distribution (PPD)* given the data,
    the labels and the image instances, which can be computed in closed form,
    as a multi-dimensional *t*-student T, and we compute the class
    log-likelihood of new samples from it.

    The PPD is given by:

    .. math::

        P(\boldsymbol{x}|\mathcal{D}) = T(\boldsymbol{x}_{ji}|\boldsymbol{
            \bar{\mu}}_k,\bar{\Sigma}_s,\bar{\nu}_s)

    where:

    .. math::

        \boldsymbol{\bar{\mu}}_k &= \frac{\kappa_k\boldsymbol{\bar{x}}_{jk}+
            \kappa_0\boldsymbol{\mu}_0}{\kappa_k+\kappa_{0}} \quad \text{where}
            \quad \kappa_k = \sum_{j=1}^{n_k}\frac{n_{jk}\kappa_{1}}
            {(n_{jk}+\kappa_{1})} \\
        \bar{\Sigma}_s &= \frac{\bar{S_s}(\bar{\kappa}_s+1)}{
            \bar{\kappa}_s\nu_s} \quad \text{where} \quad \bar{S_{s}} =
            \Sigma_{0}+\sum_{j=1}^{n_k}S_{jk} \\
        \bar{\kappa}_{s} &= \frac{\kappa_s\kappa_1}{\kappa_s+\kappa_{1}} \quad
            \text{where} \quad \kappa_s = \sum_{j=1}^{n_k}\frac{n_{jk}
            \kappa_1}{(n_{jk}+\kappa_1)}+\kappa_0 \\
        \bar{\nu}_s &= m+\sum_{j=1}^{n_k}(n_{jk}-1)-d+1

    and where :math:`n_{jk}` is the number of pixels for class *k* and instance
    *j*, :math:`S_{jk}` is the sample convariance, and
    :math:`\boldsymbol{\bar{x}}_{jk}` is the sample mean. The hyperparameters
    are :math:`\boldsymbol{\mu}_0`, :math:`\Sigma_{0}` set to the average of
    the mean of the per-class and per-instance statistics; :math:`\kappa_0=1`
    and :math:`\kappa_1=100`; and :math:`m=d+2` where *d* is the dimension of
    the data.

    If 'only_class' is False, the local prior defines an additional "outlier"
    class to detect out-of-distribution samples: if the likelihood of the prior
    is larger than any of the other classes after computing the posterior,
    the sample is considered an outlier.

    .. _Inverse Wishart: https://en.wikipedia.org/wiki/Inverse-Wishart_\
        distribution
    """

    def __init__(self, only_class=False, prior=HBMPrior()):
        """Initialize the HBM model.

        Parameters
        ----------
        only_class: bool
            do not compute outlier likelihood (classification only); default:
            False
        prior: HBMPrior
            custom prior; by default, no dimensionality reduction is used
        """
        super().__init__()
        self.prior, self.only_class, self.vtr = prior, only_class, None
        self.v_s = self.kap_s = self.mu_s = self.sig_s = self.sum_skl = None

    def fit(self, data, labels, ids=None):  # pylint: disable=arguments-differ
        """Train the HBM on data.

        Parameters
        ----------
        ids: ndarray
            image ids; if None, a dummy image id is created (not reccommended)
        """
        # pylint: disable=too-many-locals
        super().fit(data, labels)
        if ids is None:
            ids = np.zeros_like(labels)

        prior, self.vtr = self.prior.get_prior(
            data, labels, ids, self.classes)

        old_dim = data.shape[1]  # only for psi
        if self.vtr is not None:
            data = data @ self.vtr

        dim = data.shape[1]
        n_kls = len(self.classes)
        n_kls_o = n_kls if self.only_class else (n_kls + 1)  # with outliers
        psi_dofs = dim if self.only_class else old_dim

        self.kap_s = np.zeros((n_kls,))
        self.v_s = np.zeros((n_kls_o,), dtype=np.int32)
        self.mu_s = np.zeros((n_kls_o, dim))
        self.sig_s = np.zeros((dim, dim, n_kls_o))
        self.sum_skl = np.zeros((dim, dim, n_kls))

        for i, kls in enumerate(self.classes):
            in_ = labels == kls
            data_k, id_k = data[in_], ids[in_]
            uid_k = np.unique(id_k)
            n_k = len(uid_k)
            n_kl, kap = np.zeros((n_k,)), np.zeros((n_k,))
            x_kl, s_kl = np.zeros((n_k, dim)), np.zeros((dim, dim, n_k))

            for j, id_ in enumerate(uid_k):
                in_id = (id_k == id_).squeeze()
                n_kl[j] = np.sum(in_id)
                kap[j] = n_kl[j] * prior.k_1[i] / (n_kl[j] + prior.k_1[i])

                data_ki = data_k[in_id, :]
                x_kl[j, :] = np.mean(data_ki, axis=0)
                s_kl[:, :, j] = (n_kl[j] - 1) * np.cov(data_ki.T)

            sumkap = np.sum(kap) + prior.k_0[i]
            kaps = sumkap * prior.k_1[i] / (sumkap + prior.k_1[i])
            self.sum_skl[:, :, i] = np.sum(s_kl, axis=2)

            psi = prior.s_0 * (prior.wdof[i] - psi_dofs - 1) / prior.scale[i]
            self.v_s[i] = np.sum(n_kl) - n_k + prior.wdof[i] - dim + 1
            self.sig_s[:, :, i] = (psi + self.sum_skl[:, :, i]) / (
                (kaps * self.v_s[i]) / (kaps + 1))
            self.mu_s[i, :] = (np.sum(x_kl * kap[:, np.newaxis], axis=0) +
                               prior.k_0[i] * prior.mu_0) / sumkap
            self.kap_s[i] = sumkap

        if not self.only_class:  # compute also outlier likelihood
            kaps = (prior.k_0[-1] * prior.k_1[-1]) / (
                prior.k_0[-1] + prior.k_1[-1])
            self.v_s[-1] = prior.wdof[-1] - dim + 1
            self.sig_s[:, :, -1] = psi / ((kaps * self.v_s[-1]) / (kaps + 1))
            self.mu_s[-1, :] = prior.mu_0

            self.classes = np.append(self.classes, -1)  # outliers

    def _predict_proba(self, data):
        """Predicts the log-likelihood of the samples."""
        *shape, dim = data.shape
        data = data.reshape(-1, dim)  # to work on nd inputs
        if self.vtr is not None:
            data = data @ self.vtr
        dat_f = data.T.copy('F')  # Fortran order to speed up solve_triangular

        piconst = 0.5 * dim * np.log(np.pi)
        gl_pc = gammaln(np.arange(0.5, np.max(self.v_s) + dim + 0.5, 0.5))
        llh = np.zeros((data.shape[0], len(self.classes)))

        for i, _ in enumerate(self.classes):
            ch_sig = np.linalg.cholesky(self.sig_s[:, :, i])
            diff = solve_triangular(ch_sig, dat_f - self.mu_s[i:i+1].T,
                                    overwrite_b=True, check_finite=False,
                                    lower=True).T

            t_par = gl_pc[self.v_s[i] + dim - 1] - gl_pc[self.v_s[i] - 1] - \
                0.5 * dim * np.log(self.v_s[i]) - piconst - \
                np.sum(np.log(ch_sig.diagonal()))
            norm2 = np.einsum('ij,ij->i', diff, diff)  # faster than sum(x**2)
            llh[:, i] = t_par - 0.5 * (self.v_s[i] + dim) * np.log1p(
                 norm2 / self.v_s[i])

        return llh.reshape(*shape, -1)


if __name__ == '__main__':
    pass
