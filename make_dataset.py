#%%
import numpy as np
import pandas as pd

Fs = 250  # Sampling frequency
# load rest data /Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/rest_data1.csv
rest = pd.read_csv('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/rest_data1.csv')
rest.columns = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Trigger']

# load hyperventilation data
hyper1 = pd.read_csv('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/hyper_data1.csv')
hyper2 = pd.read_csv('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/hyper_data2.csv')
hyper1.columns = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Trigger']
hyper2.columns = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Trigger']


# remove the first 2 seconds of data
rest   = rest.iloc[2*Fs:]
hyper1 = hyper1.iloc[2*Fs:]
hyper2 = hyper2.iloc[2*Fs:]

# cut the data in 2 second epochs
rest_epochs = [rest.iloc[i:i+2*Fs] for i in range(0, len(rest), 2*Fs)]
hyper1_epochs = [hyper1.iloc[i:i+2*Fs] for i in range(0, len(hyper1), 2*Fs)]
hyper2_epochs = [hyper2.iloc[i:i+2*Fs] for i in range(0, len(hyper2), 2*Fs)]

# concatenate the hyperventilation epochs
hyper_epochs = hyper1_epochs[:-1] + hyper2_epochs[:-1]

# generate the condition labels for rest and hyperventilation based on the length of the epochs

y = ['rest'] * len(rest_epochs) + ['hyperventilation'] * len(hyper_epochs)
X = rest_epochs[:-1] + hyper_epochs

# save X and y to disk
np.save('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/X.npy', X)
np.save('/Users/timnaher/Documents/PhD/Projects/agitation_tvns/data/y.npy', y)



# %%
