# %%
import os

import numpy as np
import pandas as pd
from pyriemann.classification import TSclassifier
from pyriemann.estimation import Coherences
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.signal import welch

# Parameters
Fs = 250  # Sampling frequency
trial_duration = 20 * Fs  # 20 seconds in samples
trigger_start = 100  # Trigger for trial start
trigger_end = 200  # Trigger for trial end
duration = 250 * 20

# Load the data
data_path = '/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/rest_hyper_incomplete.csv'
data_path = "C:/Users/vorreuth/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/data/train/"
files = os.listdir(data_path)
data = pd.DataFrame()
for f in files:
    fdata = pd.read_csv(os.path.join(data_path, f), header=None)
    data = pd.concat([data, fdata])
# Add column names
data.columns = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'AccX', 'AccY', 'AccZ', 'GyroX',
                'GyroY', 'GyroZ', 'Trigger']


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_data(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


# Find trial start and end indices based on triggers
start_indices = data.index[data['Trigger'] == trigger_start].tolist()[:-1]
end_indices = data.index[data['Trigger'] == trigger_end].tolist()

# Epoch the data
epochs = []
labels = []
for start, end in zip(start_indices, end_indices):
    # Extract data for each trial
    epoch = data.iloc[start:(start + duration)]
    # get labels per epoch
    markers = set(epoch["Trigger"])
    if 1.0 in markers:
        label = "rest"
    elif 2.0 in markers:
        label = "hyperventilation"
    else:
        label = None
    if label:
        epochs.append(epoch)
        labels.append(label)

conditions = set(labels)
print("found conditions: {conditions} in data".format(conditions=conditions))

eeg_epochs = np.array([epoch.iloc[:, :8].values for epoch in epochs])

# compute and plot a power spectum for each epoch
spectra = {cond: [] for cond in conditions}
for epoch, label in zip(eeg_epochs, labels):
    f, Pxx = welch(epoch.T, fs=Fs, nperseg=250)
    # save the spectrum in a dict, with the corresponding condition as key
    spectra[label].append(Pxx)

# now average per condition
rest = np.mean(np.array(spectra['rest']), axis=0)
hyper = np.mean(np.array(spectra['hyperventilation']), axis=0)

# plot the average spectra
fig, ax = plt.subplots()
plt.plot(f[:40], rest.T[:40], label='rest', color='blue')
plt.plot(f[:40], hyper.T[:40], label='hyperventilation', color='red')

# now plot the grand average over all channels
grand_rest = np.mean(rest, axis=0)
grand_hyper = np.mean(hyper, axis=0)

fig, ax = plt.subplots()
plt.plot(f[:40], grand_rest[:40], label='grand rest', color='blue', linestyle='--')
plt.plot(f[:40], grand_hyper[:40], label='grand hyperventilation', color='red', linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.show()

# %% generate X and y from epochs

X = []  # This will store the one-second trials
y = []  # This will store the corresponding labels

for epoch, label in zip(eeg_epochs, labels):
    # Split each 20-second epoch into 1-second trials (250 samples each)
    for i in range(0, trial_duration, Fs):  # Fs = 250, so this moves in 1 second steps
        # filter the 1-second trial between 3 and 10 Hz

        X.append(epoch[i:i + Fs])
        y.append(label)  # Repeat the corresponding label for each 1-second trial

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# change the second and third axis
X = np.swapaxes(X, 1, 2)

# Print the shape of the resulting arrays
print("1-second trials (X) shape:", X.shape)  # Expected shape: (n_trials, 250, n_channels)
print("Labels (y) shape:", y.shape)  # Expected shape: (n_trials,)

# %% Split the data into training and testing sets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

accuracies = []

# %% Build pyriemann model to classify the data
for k in range(33):
    # covestimator = TimeDelayCovariances(delays=3, estimator='oas')
    covestimator = Coherences(window=50, overlap=0.5, fs=250)

    # Calculate the covariance matrices for each trial
    cov_train = covestimator.transform(X_train)[:, :, :, k]

    # Initialize the TSclassifier
    clf = TSclassifier()

    # Fit the classifier
    clf.fit(cov_train, y_train)

    # Calculate the covariance matrices for the test set
    cov_test = covestimator.transform(X_test)[:, :, :, k]

    # Predict the labels for the test set
    y_pred = clf.predict(cov_test)

    # Calculate the accuracy
    accuracy = np.mean(y_pred == y_test)

    # Print the accuracy
    print("Accuracy for k = {}: {:.2f}%".format(k, accuracy * 100))

    # save the accuraries in a list
    accuracies.append(accuracy)

# plot accuracy
plt.figure()
plt.plot(covestimator.freqs_, accuracies)
plt.axhline(y=0.5, color='k', linestyle='--', label="chance level")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Accuracy hyperventilation vs. rest')
plt.legend()
plt.show()