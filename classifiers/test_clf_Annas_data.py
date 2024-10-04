import os

import joblib
import numpy as np
import pandas as pd
from pyriemann.classification import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut

from agitation_tvns.src.model_pipelines import BlockKernels

Fs = 250  # Sampling frequency
trial_duration = 20 * Fs  # 20 seconds in samples
trigger_start = 100  # Trigger for trial start
trigger_end = 200  # Trigger for trial end
duration = Fs * 20

model = joblib.load('../models/model_eeg_acc_gyro.pkl')

# any files in the folder will be used; labels based on triggers (see below)
anna_data_path = "C:/Users/vorreuth/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/data/train/"
# loading in Tim's data
tim_X_path = "C:/Users/vorreuth\Downloads\X.npy"
tim_y_path = "C:/Users/vorreuth\Downloads\y.npy"
X_tim = np.load(tim_X_path)
y_tim = np.load(tim_y_path)

# reading in data
files = os.listdir(anna_data_path)
data = pd.DataFrame()
for f in files:
    fdata = pd.read_csv(os.path.join(anna_data_path, f), header=None)
    data = pd.concat([data, fdata])
# Add column namesobs
data.columns = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'AccX', 'AccY', 'AccZ', 'GyroX',
                'GyroY', 'GyroZ', 'Trigger']
start_indices = data.index[data['Trigger'] == trigger_start].tolist()[:-1]
end_indices = data.index[data['Trigger'] == trigger_end].tolist()

trigger_indices = data.index[data['Trigger'].isin([1.0, 2.0])].tolist()

# cut first 2 sec of data
data = data.iloc[2 * Fs:]

# epoch the data
epochs = []
labels = []
for start, end in zip(start_indices, end_indices):
    # for start in trigger_indices:
    # Extract data for each trial
    epoch = data.iloc[start:(start + duration)]
    markers = set(epoch["Trigger"])
    # get labels per epoch
    if 1.0 in markers:
        label = "rest"
    elif 2.0 in markers:
        label = "hyperventilation"
    else:
        label = None
    if label:
        trigger = epoch.index[epoch['Trigger'].isin([1.0, 2.0])].tolist()[0]
        epoch = data.iloc[trigger:(trigger + duration)]
        epoch = epoch.drop(columns=["Trigger"])
        epoch = [epoch.iloc[i:i + 2 * Fs] for i in range(0, len(epoch), 2 * Fs)]
        epochs.append(epoch)
        labels.append([label] * len(epoch))

labels = [x for xs in labels for x in xs]
conditions = set(labels)
print("found conditions: {conditions} in data".format(conditions=conditions))

# reduce to only EEG data
# eeg_epochs = np.array([epoch.iloc[:, :8].values for epoch in epochs])

# generate X and y from epochs
X = np.array([np.array(epoch).T for epoch in epochs])
X = X.reshape(X.shape[0] * X.shape[-1], 14, 500)
y = np.array(labels)

assert X_tim.shape[1:] == X.shape[1:]
# concatenate data from train folder with Tim's data
X = np.concatenate([X, X_tim], axis=0)
y = np.concatenate([y, y_tim], axis=0)

# Print the shape of the resulting arrays
print("trials (X) shape:", X.shape)
print("labels (y) shape:", y.shape)

# save concatenated X and y
np.save('X_all.npy', X)
np.save('y_all.npy', y)

kernel_functions = ["linear", "poly", "polynomial", "rbf", "laplacian", "cosine"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
model = Pipeline(
    [("block_kernels", BlockKernels(
        block_size=[8, 3, 3], metric=["rbf", "corr", 'rbf'], shrinkage=[0.1, 0.1, 0.1], ),), ("classifier", SVC()), ]
    )
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy for this iteration
accuracy = np.mean(y_pred == y_test)

print(f'train-test-split accuracy:{accuracy}')


# Leave-One-Out Cross-Validation
# loo = LeaveOneOut()
# accuracies = []
# for train_index, test_index in loo.split(X):
#     # Split data
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     model = Pipeline(
#         [("block_kernels", BlockKernels(
#             block_size=[8, 3, 3], metric=["rbf", "corr", 'rbf'], shrinkage=[0.1, 0.1, 0.1], ),),
#          ("classifier", SVC()), ]
#         )
#     model.fit(X_train, y_train)
#
#     # Predict on test data
#     y_pred = model.predict(X_test)
#
#     # Calculate accuracy for this iteration
#     accuracy = np.mean(y_pred == y_test)
#     accuracies.append(accuracy)
#
# # calculate accuracy
# mean_accuracy = np.mean(accuracies)
# print(f'loo accuracy:{mean_accuracy}')