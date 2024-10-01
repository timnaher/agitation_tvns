#%%
import numpy as np
import pylsl
import time

# this block will got src
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from pyriemann.utils.covariance import covariances
from pyriemann.estimation import Shrinkage
from pyriemann.classification import SVC
from pyriemann.utils.covariance import cov_est_functions
kernel_functions = [
    "linear", "poly", "polynomial", "rbf", "laplacian", "cosine"
]
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

# enf of block
# load the model at C:\Users\TVN Sleep\Documents\gtec\agitation_tvns\models\model_eeg_acc_gyro.pkl
model_path = Path("C:/Users/TVN Sleep/Documents/gtec/agitation_tvns/models/model_eeg_acc_gyro.pkl")
model = joblib.load(model_path)

# Define the stream name
stream_name = "UnicornRecorderLSLStream"

# Resolve the LSL stream
print(f"Looking for an LSL stream named '{stream_name}'...")
streams = pylsl.resolve_byprop('name', stream_name)

if len(streams) == 0:
    raise RuntimeError(f"No stream with the name '{stream_name}' found.")

# Connect to the stream
inlet = pylsl.StreamInlet(streams[0])
info = inlet.info()

# Get information about the stream
sampling_rate = int(info.nominal_srate())
num_channels = info.channel_count()

print(f"Connected to stream: {stream_name}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Number of channels: {num_channels}")

# 1-second buffer size based on the sampling rate
buffer_size = sampling_rate*2

# Start collecting data in "-second buffers
try:
    while True:
        # Create an empty buffer to store 2 second of data
        data_buffer = np.zeros((buffer_size, num_channels))

        # Collect data for 2 second
        for i in range(buffer_size):
            sample, timestamp = inlet.pull_sample()
            data_buffer[i, :] = sample

        # After 1 second, the buffer is ready for classification
        
        # CLASSIFICATION WILL HAPPEN HERE
        X = data_buffer.T
        y_pred = model.predict(X[None,:-2,:])
        print(y_pred)

        # BASED ON CLASSIFICATION, HUE API WILL BE CALLED HERE

        # BASED ON CLASSIFICATION, tVNS API WILL BE CALLED HERE

        # short sleep
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Data collection stopped.")

#%%