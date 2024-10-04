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

# TODO: NOTE FOR ANNA AND LISA:
# if this import does not work, make sure that you have installed the packge
# by running `pip install -e .` in the root directory of the project
# also if it does not work at all, then just define the BlockKernel model
# in this file. 
from agitation_tvns.src.model_pipelines import BlockKernels

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
