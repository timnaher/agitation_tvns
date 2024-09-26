#%%
import UnicornPy
import numpy as np
import matplotlib.pyplot as plt

# Discover available devices
available_devices = UnicornPy.GetAvailableDevices(True)  # True to search only for paired devices
if len(available_devices) == 0:
    raise Exception("No Unicorn devices found")

# Connect to the first available device
device_name = available_devices[0]
unicorn = UnicornPy.Unicorn(device_name)
print(f"Connected to device: {device_name}")

# Get device configuration
config = unicorn.GetConfiguration()
num_channels = unicorn.GetNumberOfAcquiredChannels()

# Set up the plot
plt.ion()  # Interactive mode on for dynamic updating
fig, ax = plt.subplots(num_channels, 1, figsize=(10, 8))

# Start acquisition in measurement mode (False disables test signal mode)
unicorn.StartAcquisition(False)
print("Data acquisition started.")

# Define the buffer to hold 1 second of data (250 samples per second)
sampling_rate = 250  # Unicorn BCI samples at 250 Hz
num_scans = sampling_rate  # 1 second buffer
buffer_length = num_scans * num_channels
data_buffer = bytearray(buffer_length * 4)  # 4 bytes per float32

try:
    while True:
        # Read 1 second of data (250 scans)
        unicorn.GetData(num_scans, data_buffer, buffer_length)

        # Convert the data buffer into a NumPy array for easier manipulation
        data_array = np.frombuffer(data_buffer, dtype=np.float32).reshape((num_scans, num_channels))

        # Update the plot with the new data
        for i in range(num_channels):
            ax[i].cla()  # Clear the previous plot
            ax[i].plot(data_array[:, i])
            ax[i].set_title(f"Channel {i+1}")
            ax[i].set_xlim([0, num_scans])

        plt.pause(0.001)  # Small pause to allow for plot update

except KeyboardInterrupt:
    # Stop the acquisition when the user interrupts the script (Ctrl+C)
    unicorn.StopAcquisition()
    print("Data acquisition stopped.")
    plt.ioff()  # Turn off interactive mode
    plt.show()

#%%