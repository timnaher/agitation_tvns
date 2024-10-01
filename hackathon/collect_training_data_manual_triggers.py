import time

import keyboard
import socket


def on_key_event(event):
    if event.event_type == keyboard.KEY_DOWN:
        sendBytes = ''
        if event.name == '1' or event.name == 'r':
            sendBytes = b"1"
        if event.name == '2' or event.name == 'h':
            sendBytes = b"2"
        if len(sendBytes) > 0:
            print('key: ' + event.name + ' sending: ' + str(sendBytes))
            socket.sendto(b"100", endPoint)
            time.sleep(1.5)
            socket.sendto(sendBytes, endPoint)
            time.sleep(5)
            socket.sendto(b"200", endPoint)


# Initialize socket
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
endPoint = ("127.0.0.1", 1000)

keyboard.on_press(on_key_event)
keyboard.wait('esc')
