#%%
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import os
from pathlib import Path
import urllib.request
from scipy.signal import welch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from pyriemann.utils.covariance import covariances
from pyriemann.estimation import Shrinkage
from pyriemann.classification import SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.covariance import cov_est_functions
from sklearn.metrics import accuracy_score


#from agitation_tvns.model_pipelines import BlockKernels



# load the X and y data
X = np.load('/Users/lisa-mariebastian/Documents/projects/heckathon_unicorn/X.npy', allow_pickle=True)
y = np.load('/Users/lisa-mariebastian/Documents/projects/heckathon_unicorn/y.npy', allow_pickle=True)


# in X, only keep the EEG data
eeg_data = np.array([epoch[:8,:] for epoch in X])


# Frequency bands: Delta, Theta, Alpha, Beta
bands = {'delta': (1, 3), 'theta': (4, 7), 'alpha': (8, 12), 'beta': (16, 20)}
fs = 250

# Function to extract band power from PSD
def bandpower(psd, freqs, band):
    band_freqs = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[band_freqs])

# Function to compute features for all trials
def compute_features(eeg_data, fs, bands):
    num_trials, num_channels, _ = eeg_data.shape
    features = np.zeros((num_trials, num_channels * len(bands)))

    for trial in range(num_trials):
        trial_features = []
        for ch in range(num_channels):
            # Compute power spectral density (Welch's method)
            freqs, psd = welch(eeg_data[trial, ch, :], fs, nperseg=250, noverlap=200)
            
            # Extract power in each frequency band
            for band_name, band_range in bands.items():
                power = bandpower(psd, freqs, band_range)
                trial_features.append(power)

        features[trial, :] = np.array(trial_features).flatten()
    return features

# Compute features for rest and hyperventilation states
X = compute_features(eeg_data, fs, bands)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict and evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')


#%%

# save the trained model to disk
#import joblib
#joblib.dump(model, '/Users/lisa-mariebastian/Documents/projects/heckathon_unicorn/data/model_freqbands.pkl')


