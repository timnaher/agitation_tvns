#%%
import numpy as np
import pylsl
import time

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
buffer_size = sampling_rate

# Start collecting data in 1-second buffers
try:
    while True:
        # Create an empty buffer to store 1 second of data
        data_buffer = np.zeros((buffer_size, num_channels))

        # Collect data for 1 second
        for i in range(buffer_size):
            sample, timestamp = inlet.pull_sample()
            data_buffer[i, :] = sample

        # After 1 second, the buffer is ready for classification
        print(f"Collected 1 second of data with shape: {data_buffer.shape}")
        
        # CLASSIFICATION WILL HAPPEN HERE

        # BASED ON CLASSIFICATION, HUE API WILL BE CALLED HERE

        # BASED ON CLASSIFICATION, tVNS API WILL BE CALLED HERE


        # short sleep
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Data collection stopped.")

#%%