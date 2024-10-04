###########################################################################
# tVNS Manager 2.1.0.0
# python 3.12 (64-bit)
# Visual Studio Code: 1.89.0
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


def start_tvns(url, endPoint):
    response = requests.post(url, data='manualSwitch')
    response = requests.post(url, data='startTreatment')
    socket.sendto(b"100", endPoint)
    print(f'tVNS start | EEG trigger 100')
    return True


def stop_tvns(url, endPoint):
    response = requests.post(url, data='stopTreatment')
    socket.sendto(b"200", endPoint)
    print(f'tVNS stop | EEG trigger 200')


def send_stimulation(url, intensity, endPoint, stim_trigger=3, duration=5):
    # response = requests.post(url, data='startStimulation')
    response = requests.post(url, data=f'intensity {intensity}')  # set the intensity in the decvice
    socket.sendto(b"%d" % stim_trigger, endPoint)
    print(f'tVNS stimulation | EEG trigger {stim_trigger}')
    time.sleep(duration)
    response = requests.post(url, data='stopStimulation')

def customise_params(
        url,
        minIntensity=100,
        maxIntensity=500,
        impulseDuration=250,
        frequency=25,
        stimulationDuration=28,
        pauseDuration=32
        ):
    # set stimulation parameters
    response = requests.post(
        url, data=f'minIntensity={minIntensity}&maxIntensity={maxIntensity}&impulseDuration'
                  f'={impulseDuration}&frequency={frequency}&stimulationDuration={stimulationDuration}&pauseDuration={pauseDuration}'
        )

##################
# EXAMPLE USAGE
##################

# set initial config
# tVNS_params = {
#     "minIntensity": 100,
#     "maxIntensity": 5000,
#     "impulseDuration": 250,
#     "frequency": 25,
#     "stimulationDuration": 28,
#     "pauseDuration": 32
#     }
# tVNS_intensity = 1200
#
# try:
#     ## setup ##
#     # EEG trigger setup
#     socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     endPoint = ("127.0.0.1", 1000)
#     print(f"EEG trigger endPoint: {endPoint}")
#     # tVNS URL
#     url = 'http://localhost:51523/tvnsmanager/'
#     print(f"tVNS URL: {url}")
#     customise_params(url, **tVNS_params)
#
#     ## main functions ##
#     start_tvns(url, endPoint)
#     time.sleep(5)
#     send_stimulation(url, tVNS_intensity, endPoint)
#     time.sleep(5)
#     stop_tvns(url, endPoint)
#
# except requests.exceptions.ConnectionError as e:
#     print("tVNS device connection not found")
#     print(e)
# except Exception as e:
#     print("an error occurred")
#     print(e)
#

