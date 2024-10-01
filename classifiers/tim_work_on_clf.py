#%%
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from pyriemann.utils.covariance import covariances
from pyriemann.estimation import Shrinkage
from pyriemann.classification import SVC
from pyriemann.utils.covariance import cov_est_functions

#TODO: move models to src

# load the X and y data
X = np.load('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/X.npy', allow_pickle=True)
y = np.load('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/y.npy', allow_pickle=True)


# in X, only keep the EEG data

# make a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# shape is epochs x time x channels

kernel_functions = [
    "linear", "poly", "polynomial", "rbf", "laplacian", "cosine"
]


#%% Define some models

class BlockKernels(BaseEstimator, TransformerMixin):
    """Estimation of block kernel or covariance matrices with
    customizable metrics and shrinkage.

    Perform a block matrix estimation for each given time series,
    computing either kernel matrices or covariance matrices for
    each block based on the specified metrics. Optionally apply
    shrinkage to each block's matrix.

    Parameters
    ----------
    block_size : int | list of int
        Sizes of individual blocks given as int for same-size blocks,
        or list for varying block sizes.
    metric : string | list of string, default='linear'
        The metric(s) to use when computing matrices between channels.
        For kernel matrices, supported metrics are those from
        ``pairwise_kernels``: 'linear', 'poly', 'polynomial',
        'rbf', 'laplacian', 'cosine', etc. For covariance matrices,
        supported estimators are those from pyRiemann:
        'scm', 'lwf', 'oas', 'mcd', etc.
        If a list is provided, it must match the number of blocks.
    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by
        breaking down the pairwise matrix into ``n_jobs`` even
        slices and computing them in parallel.
    shrinkage : float | list of float, default=0
        Shrinkage parameter(s) to regularize each block's matrix.
        If a single float is provided, it is applied to all blocks.
        If a list is provided, it must match the number of blocks.
    **kwds : dict
        Any further parameters are passed directly to the kernel function(s)
        or covariance estimator(s).

    See Also
    --------
    BlockCovariances
    Kernels
    Shrinkage
    """

    def __init__(
        self, block_size, metric="linear", n_jobs=None, shrinkage=0, **kwds
    ):
        self.block_size = block_size
        self.metric = metric
        self.n_jobs = n_jobs
        self.shrinkage = shrinkage
        self.kwds = kwds

    def fit(self, X, y=None):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Multi-channel time series.
        y : None
            Not used, here for compatibility with scikit-learn API.

        Returns
        -------
        self : BlockKernels instance
            The BlockKernels instance.
        """
        return self

    def transform(self, X):
        """Estimate block kernel or covariance matrices
        with optional shrinkage.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_channels, n_times)
            Multi-channel time series.

        Returns
        -------
        M : ndarray, shape (n_samples, n_channels, n_channels)
            Block matrices (kernel or covariance matrices).
        """
        n_samples, n_channels, n_times = X.shape

        blocks = self._check_block_size(
            self.block_size,
            n_channels,
        )
        n_blocks = len(blocks)

        # Handle metric parameter
        if isinstance(self.metric, str):
            metrics = [self.metric] * n_blocks
        elif isinstance(self.metric, list):
            if len(self.metric) != n_blocks:
                raise ValueError(
                    f"Length of metric list ({len(self.metric)}) must"
                    f"match number of blocks ({n_blocks})"
                )
            metrics = self.metric
        else:
            raise ValueError(
                "Parameter metric must be a string or a list of strings."
            )

        # Handle shrinkage parameter
        if isinstance(self.shrinkage, (float, int)):
            shrinkages = [self.shrinkage] * n_blocks
        elif isinstance(self.shrinkage, list):
            if len(self.shrinkage) != n_blocks:
                raise ValueError(
                    f"Length of shrinkage list ({len(self.shrinkage)})"
                    f"must match number of blocks ({n_blocks})"
                )
            shrinkages = self.shrinkage
        else:
            raise ValueError(
                "Parameter shrinkage must be a float, or a list of floats."
            )

        M_matrices = []

        for i in range(n_samples):
            start = 0
            M_blocks = []
            for idx, (block_size, metric, shrinkage_value) in enumerate(
                zip(blocks, metrics, shrinkages)
            ):
                end = start + block_size
                # Extract the block of channels
                X_block = X[i, start:end, :]  # shape: (block_size, n_times)

                # Compute the matrix for this block
                if metric in kernel_functions:
                    # Compute kernel matrix
                    M_block = pairwise_kernels(
                        X_block, metric=metric, n_jobs=self.n_jobs, **self.kwds
                    )
                elif metric in cov_est_functions.keys():
                    # Compute covariance matrix
                    X_block_reshaped = X_block[np.newaxis, :, :]
                    M_block = covariances(
                        X_block_reshaped, estimator=metric, **self.kwds
                    )[0]
                else:
                    raise ValueError(
                        f"Metric '{metric}' is not recognized"
                        " as a kernel metric or a covariance estimator."
                    )

                # Apply shrinkage if specified
                if shrinkage_value != 0:
                    M_block_reshaped = M_block[np.newaxis, :, :]
                    shr = Shrinkage(shrinkage=shrinkage_value)
                    M_block = shr.fit_transform(M_block_reshaped)[0]

                M_blocks.append(M_block)
                start = end

            # Create the block diagonal matrix
            M_full = self._block_diag(M_blocks)
            M_matrices.append(M_full)

        return np.array(M_matrices)

    @staticmethod
    def _block_diag(matrices):
        """Construct a block diagonal matrix from a list of square matrices."""
        shapes = [m.shape[0] for m in matrices]
        total_size = sum(shapes)
        result = np.zeros((total_size, total_size), dtype=matrices[0].dtype)
        start = 0
        for m in matrices:
            end = start + m.shape[0]
            result[start:end, start:end] = m
            start = end
        return result

    @staticmethod
    def _check_block_size(block_size, n_channels):
        """Check validity of parameter block_size"""
        if isinstance(block_size, int):
            if n_channels % block_size != 0:
                raise ValueError(
                    f"Number of channels ({n_channels}) must be "
                    f"divisible by block size ({block_size})"
                )
            n_blocks = n_channels // block_size
            blocks = [block_size] * n_blocks

        elif isinstance(block_size, (list, np.ndarray)):
            if np.sum(block_size) != n_channels:
                raise ValueError(
                    "Sum of individual block sizes must match "
                    f"number of channels ({n_channels})"
                )
            blocks = block_size

        else:
            raise ValueError("Parameter block_size must be int or list.")

        return blocks


#%%

model = Pipeline(
    [
        (
            "block_kernels",
            BlockKernels(
                block_size=[8, 3, 3],
                metric=["rbf", "corr", 'rbf'],
                shrinkage=[0.1, 0.1, 0.1],
            ),
        ),
        ("classifier", SVC()),
    ]
)


# fit on training data
model.fit(X_train, y_train)

# predict on test data
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(accuracy)


# save the trained model to disk
import joblib
joblib.dump(model, '/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/model_eeg_acc_gyro.pkl')

# %%
