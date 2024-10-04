###########################################################################
# tVNS Manager 2.1.0.0 
# 
# Example for Python
#
# tVNS Technologies GmbH
# 2024-06-28
# Tobias Jeglorz
#
# python 3.12 (64-bit)
# Visual Studio Code: 1.89.0
#
# first install and start the tVNS Manager
# bond and pair the tVNS R device with the tVNS Manager
# Run this application
############################################################################
'''
You can now control your device using HTTP POST requests.
Send a POST request to http://localhost:51523/tvnsmanager/ with body
'manualSwitch' or 'automaticSwitch' to initialise the connection with your client.
In manual switch mode, the on- and off-phases can be controlled using the
POST requests 'startStimulation' and 'stopStimulation'.
In automatic switch mode, the device configuration is used to switch between
the on- and off-phases.
Send a POST request with body 'startTreatment' to begin a treatment for
logging purposes.
When a treatment is complete, send POST request with body 'stopTreatment'
to end a treatment and record a log.
'''

import socket
import time
import requests


# to start
def start_tvns(url, endPoint):
    response = requests.post(url, data='manualSwitch')
    response = requests.post(url, data='startTreatment')
    response = requests.post(url, data='startStimulation')
    socket.sendto(b"100", endPoint)
    print(f'tVNS start | EEG trigger 100')


# to stop
def stop_tvns(url, endPoint):
    response = requests.post(url, data='stopStimulation')
    response = requests.post(url, data='stopTreatment')
    socket.sendto(b"200", endPoint)
    print(f'tVNS stop | EEG trigger 200')


# send stimulation
def send_stimulation(url, intensity, endPoint, stim_trigger=3):
    response = requests.post(url, data=f'intensity {intensity}')  # set the intensity in the decvice
    socket.sendto(b"%d" % stim_trigger, endPoint)
    print(f'tVNS stimulation | EEG trigger {stim_trigger}')

# set stimulation parameters
def customise_params(
        url,
        minIntensity=100,
        maxIntensity=500,
        impulseDuration=250,
        frequency=25,
        stimulationDuration=28,
        pauseDuration=32
        ):
    response = requests.post(
        url, data=f'minIntensity={minIntensity}&maxIntensity={maxIntensity}&impulseDuration'
                  f'={impulseDuration}&frequency={frequency}&stimulationDuration={stimulationDuration}&pauseDuration={pauseDuration}'
        )

##################
# EXAMPLE USAGE
##################

# set initial config
tVNS_params = {
    "minIntensity": 100,
    "maxIntensity": 5000,
    "impulseDuration": 250,
    "frequency": 25,
    "stimulationDuration": 28,
    "pauseDuration": 32
    }
tVNS_intensity = 1200

try:
    # tVNS URL
    url = 'http://localhost:51523/tvnsmanager/'
    print(f"tVNS URL: {url}")
    # EEG trigger setup
    socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    endPoint = ("127.0.0.1", 1000)
    print(f"EEG trigger endPoint: {endPoint}")
except requests.exceptions.ConnectionError as e:
    print("tVNS device connection not found")
    print(e)
    
# set parameters
customise_params(url, **tVNS_params)

# start treatment + stimulation
start_tvns(url, endPoint)

time.sleep(5)

send_stimulation(url, tVNS_intensity, endPoint)

time.sleep(5)

# stop treatment + stimulation
stop_tvns(url, endPoint)
