#%%
import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
from scipy.signal import firwin

# Resolve the LSL stream from the Unicorn headset
print("Looking for a Unicorn EEG stream...")
streams = resolve_stream('type', 'EEG')

# Create an inlet to read from the stream
inlet = StreamInlet(streams[0])

# Set up the plot
nchan = 16     # Total channels
fsample = 250  # Sampling frequency (Hz)
time_window = 5  # Seconds of data to display
buffer_size = int(fsample * time_window)

# FIR bandpass filter parameters
lowcut = 1.0    # Low cutoff frequency (Hz)
highcut = 30.0  # High cutoff frequency (Hz)
filter_order = 101  # Filter order (number of taps)

# Design the FIR filter
nyquist = 0.5 * fsample
low = lowcut / nyquist
high = highcut / nyquist
fir_coeff = firwin(filter_order, [low, high], pass_zero=False)

# Initialize buffers for filtering (one per EEG channel)
filter_buffers = [np.zeros(filter_order) for _ in range(8)]  # For 8 EEG channels

# Initialize data buffers for plotting
eeg_data = np.zeros((buffer_size, 8))
eeg_filtered = np.zeros((buffer_size, 8))
accel_data = np.zeros((buffer_size, 3))

# Create subplots for EEG and accelerometer data
fig, (ax_eeg, ax_accel) = plt.subplots(2, 1, figsize=(12, 8))

# EEG plot for 8 channels
lines_eeg = [ax_eeg.plot(np.arange(buffer_size), eeg_filtered[:, ch])[0] for ch in range(8)]
ax_eeg.set_xlim(0, buffer_size)
ax_eeg.set_title('Filtered EEG Data (1-30 Hz)')
ax_eeg.set_xlabel('Samples')
ax_eeg.set_ylabel('Amplitude (µV)')

# Accelerometer plot for 3 axes
lines_accel = [ax_accel.plot(np.arange(buffer_size), accel_data[:, ch])[0] for ch in range(3)]
ax_accel.set_xlim(0, buffer_size)
ax_accel.set_title('Accelerometer Data')
ax_accel.set_xlabel('Samples')
ax_accel.set_ylabel('Acceleration (g)')

# Function to update the plot
def update_plot():
    # Update EEG plot
    for ch in range(8):
        lines_eeg[ch].set_ydata(eeg_filtered[:, ch])

    # Dynamically adjust y-limits for EEG plot
    eeg_max = np.max(eeg_filtered)
    eeg_min = np.min(eeg_filtered)
    eeg_range = eeg_max - eeg_min
    if eeg_range == 0:
        eeg_range = 1  # Avoid division by zero
    ax_eeg.set_ylim(eeg_min - 0.1 * eeg_range, eeg_max + 0.1 * eeg_range)

    # Center the accelerometer data
    accel_data_centered = accel_data - np.mean(accel_data, axis=0)

    # Update Accelerometer plot
    for ch in range(3):
        lines_accel[ch].set_ydata(accel_data_centered[:, ch])

    # Dynamically adjust y-limits for Accelerometer plot
    accel_max = np.max(accel_data_centered)
    accel_min = np.min(accel_data_centered)
    accel_range = accel_max - accel_min
    if accel_range == 0:
        accel_range = 1  # Avoid division by zero
    ax_accel.set_ylim(accel_min - 0.1 * accel_range, accel_max + 0.1 * accel_range)

    # Redraw the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

# Start plotting in real-time
plt.show(block=False)
print("Starting real-time plot...")

try:
    while True:
        # Get the new sample from the LSL stream
        sample, timestamp = inlet.pull_sample()

        # Shift data buffers and append new sample
        eeg_data = np.roll(eeg_data, -1, axis=0)
        eeg_data[-1, :] = sample[0:8]  # EEG channels 0-7

        accel_data = np.roll(accel_data, -1, axis=0)
        accel_data[-1, :] = sample[8:11]  # Accelerometer channels 8-10

        # Center the EEG data by subtracting the mean of the buffer
        eeg_sample_centered = eeg_data[-1, :] - np.mean(eeg_data, axis=0)

        # Apply the FIR filter to the current sample for each channel
        for ch in range(8):
            # Update the filter buffer for the channel
            filter_buffers[ch][:-1] = filter_buffers[ch][1:]  # Shift left
            filter_buffers[ch][-1] = eeg_sample_centered[ch]   # Insert new sample

            # Compute the filtered sample (dot product with reversed coefficients)
            filtered_sample = np.dot(fir_coeff, filter_buffers[ch][::-1])

            # Update the filtered data buffer
            eeg_filtered = np.roll(eeg_filtered, -1, axis=0)
            eeg_filtered[-1, ch] = filtered_sample

        # Update the plot
        update_plot()

        # Small delay to match the sampling rate
        time.sleep(1.0 / fsample)

except KeyboardInterrupt:
    print("Stopping the real-time plot.")
    plt.close()
# %%
