import pandas as pd
import joblib
import os
import numpy as np
from agitation_tvns.src.model_pipelines import BlockKernels

Fs = 250  # Sampling frequency
trial_duration = 20 * Fs  # 20 seconds in samples
trigger_start = 100  # Trigger for trial start
trigger_end = 200  # Trigger for trial end
duration = Fs * 20

model = joblib.load('../models/model_eeg_acc_gyro.pkl')

# any files in the folder will be used; labels based on triggers (see below)
anna_data_path = "C:/Users/vorreuth/Documents/gtec/Unicorn Suite/Hybrid Black/Unicorn Recorder/data/train/"
# loading in X.npy for shape verification
tim_data_path = "C:/Users/vorreuth\Downloads\X.npy"

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


# cut first 2 sec of data
data = data.iloc[2*Fs:]

# epoch the data
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
        epoch = epoch.drop(columns=["Trigger"])
        epoch = [epoch.iloc[i:i + 2 * Fs] for i in range(0, len(epoch), 2 * Fs)]
        epochs.append(epoch)
        labels.append([label]*len(epoch))

labels = [x for xs in labels for x in xs]
conditions = set(labels)
print("found conditions: {conditions} in data".format(conditions=conditions))

# reduce to only EEG data
# eeg_epochs = np.array([epoch.iloc[:, :8].values for epoch in epochs])

# generate X and y from epochs
X = np.array([np.array(epoch).T for epoch in epochs])
X = X.reshape(X.shape[0]*X.shape[-1], 14, 500)
y = np.array(labels)


# Print the shape of the resulting arrays
print("trials (X) shape:", X.shape)  # Expected shape: (n_trials, 250, n_channels)
print("labels (y) shape:", y.shape)  # Expected shape: (n_trials,)

X_tim = np.load(tim_data_path)
assert X_tim.shape[1:] == X.shape[1:]

y_pred = model.predict(X)

# calculate accuracy
accuracy = np.mean(y_pred == y)
print(f'accuracy:{accuracy}')
