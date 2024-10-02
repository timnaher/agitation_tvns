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
    #fig, ax = plt.subplots()
    #plt.plot(f[:40], Pxx.T[:40])
    # add a title with the current condition
    #ax.set_title(conditions[k])

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

# Split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# %% Build pyriemann model to classify the data
for k in range(33):
    # Calculate the covariance matrices for each trial on the training set
    cov_train_full = covestimator.transform(X_train_full)[:, :, :, k]

    # Initialize and fit the TSclassifier
    clf = TSclassifier()
    clf.fit(cov_train_full, y_train_full)

    # Calculate the covariance matrices for the validation set
    cov_val = covestimator.transform(X_val)[:, :, :, k]

    # Predict the labels for the validation set
    y_val_pred = clf.predict(cov_val)

    # Calculate the validation accuracy
    val_accuracy = np.mean(y_val_pred == y_val)

    # Save the validation accuracies in a list
    val_accuracies.append(val_accuracy)

    # Print validation accuracy for each frequency
    print("Validation Accuracy for k = {}: {:.2f}%".format(k, val_accuracy * 100))




# plot the data
plt.plot(covestimator.freqs_, val_accuracies)

# vertical x line at 50

# add a horizontal line at 0.5 to show the chance level
plt.axhline(y=0.5, color='k', linestyle='--')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Accuracy Hyperventilation vs Rest')


# identify the 2 best frequencies with the hightest accuracy
best_val_freqs = np.argsort(val_accuracies)[-3:]

# draw a vertical line at the best frequencies
plt.plot(covestimator.freqs_[best_val_freqs], [0.5, 0.5, 0.5], 'ro')


# %% Create a block diagonal matrix from the best frequencies
from scipy.linalg import block_diag

# Function to create a block diagonal matrix from selected frequency covariance matrices
def create_block_diagonal_matrix(cov_matrices, best_freqs):
    block_matrices = []
    for i in best_freqs:
        # Extract the covariance matrix corresponding to each of the best frequencies
        block_matrices.append(cov_matrices[:, :, :, i])

    # Stack the covariance matrices along the diagonal for each trial
    block_diag_cov = np.array([block_diag(*blocks) for blocks in zip(*block_matrices)])

    return block_diag_cov

# Apply this to the training set
cov_train_full_all_freqs = covestimator.transform(X_train_full)
block_diag_cov_train = create_block_diagonal_matrix(cov_train_full_all_freqs, best_val_freqs)

# Apply the same to the validation and test sets
cov_val_all_freqs = covestimator.transform(X_val)
block_diag_cov_val = create_block_diagonal_matrix(cov_val_all_freqs, best_val_freqs)

cov_test_all_freqs = covestimator.transform(X_test)
block_diag_cov_test = create_block_diagonal_matrix(cov_test_all_freqs, best_val_freqs)

print("Block diagonal covariance matrix shape (training):", block_diag_cov_train.shape)

# %% Build an estimator using the block diagonal matrices
from pyriemann.classification import TSclassifier, FgMDM, SVC

# Use a simple classifier like MDM (Mean of Riemannian distances) for the block diagonal covariance matrices
clf_block_diag = SVC()

# Fit the classifier on the training set
clf_block_diag.fit(block_diag_cov_train, y_train_full)

# Evaluate on the validation set
y_val_pred_block_diag = clf_block_diag.predict(block_diag_cov_val)
val_accuracy_block_diag = np.mean(y_val_pred_block_diag == y_val)

print("Validation Accuracy with Block Diagonal Covariance Matrices: {:.2f}%".format(val_accuracy_block_diag * 100))

# Evaluate on the test set
y_test_pred_block_diag = clf_block_diag.predict(block_diag_cov_test)
test_accuracy_block_diag = np.mean(y_test_pred_block_diag == y_test)

print("Test Accuracy with Block Diagonal Covariance Matrices: {:.2f}%".format(test_accuracy_block_diag * 100))

# %%
