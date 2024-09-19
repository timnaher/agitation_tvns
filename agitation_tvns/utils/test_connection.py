import serial
import time

import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for port in ports:
    print(port.device)

device = '/dev/cu.UN-20230806'
start_acq = [0x61, 0x7C, 0x87]

try:
    s = serial.Serial(
    port=device,
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=5
)
    print("Connected to serial port " + device)
except:
    raise RuntimeError("Cannot connect to serial port " + device)
s.timeout = 5  # Increase timeout to 5 seconds
s.write(bytes(start_acq))
s.flush()
time.sleep(1)  # Allow some delay for processing
response = s.read(3)  # Try reading more bytes, in case the response is larger
print(f"Response: {response}")
