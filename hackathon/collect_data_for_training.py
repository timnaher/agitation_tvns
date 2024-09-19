
import psychopy.visual
import psychopy.event
import time
import random
import numpy as np

# Global variables
win = None # Global variable for window (Initialized in main)
fixation = None # Global variable for fixation cross (Initialized in main)
bg_color = [0, 0, 0]
win_w = 800
win_h = 600
refresh_rate = 165.0 # Monitor refresh rate (CRITICAL FOR TIMING)

#========================================================
# High Level Functions
#========================================================
def Paradigm(trials):
    # Initialize fixation cross
    fixation = psychopy.visual.ShapeStim(
                win=win,
                units='pix',
                size=50,
                fillColor=[1, 1, 1],
                lineColor=[1, 1, 1],
                lineWidth=1,
                vertices='cross',
                name='off',
                pos=[0, 0]
            )
    
    # Initialize text (for instructions & results)
    text = psychopy.visual.TextStim(win,
                    'TEST TEXT', font='Open Sans', units='pix', 
                    pos=(0, 0), alignText='center',
                    height=36, color=[1, 1, 1]
                    )

    for trial in trials:
        # 1000 ms direction instruction (blink or no blink)
        text.text = trial
        for frame in range(MsToFrames(1500, refresh_rate)):
            text.draw()
            win.flip()
            
        # 500ms blank screen
        for frame in range(MsToFrames(500, refresh_rate)):
            win.flip()
    
        # 3000ms fixation cross (doing MI)
        for frame in range(MsToFrames(5000, refresh_rate)):
            fixation.draw()
            win.flip()
            
        # 500ms blank screen
        for frame in range(MsToFrames(500, refresh_rate)):
            win.flip()

        # 1000ms blank screen
        for frame in range(MsToFrames(1000, refresh_rate)):
            win.flip()
    
    print("Paradigm complete")

def MsToFrames(ms, fs):
    dt = 1000 / fs
    return np.round(ms / dt).astype(int)

if __name__ == "__main__":
    # Set random seed
    random.seed()
    
    # Create PsychoPy window
    win = psychopy.visual.Window(
        screen=0,
        size=[win_w, win_h],
        units="pix",
        fullscr=False,
        color=bg_color,
        gammaErrorPolicy="ignore"
    )
    
    # Wait a second for the window to settle down
    time.sleep(1)

    # Define the trials: 10 "blink" and 10 "no blink"
    trials = ['normal', 'hyperventilation'] * 10
    random.shuffle(trials)

    # Run through paradigm
    Paradigm(trials)

    # Close the window
    win.close()