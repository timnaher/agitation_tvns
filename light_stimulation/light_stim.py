import time

from phue import Bridge


# Function to make the light pulsate
def pulsate_light(
        light, breath_in_time, breath_out_time, min_brightness, max_brightness, pause_time, duration_light_stim
        ):
    try:
        counter = 0
        while counter < duration_light_stim:  # Continuous breathing loop until stopped
            # Breath out (decrease brightness)
            for brightness in range(max_brightness, min_brightness - 1, -breath_in_time-1):
                light.brightness = brightness
                time.sleep(breath_in_time / (max_brightness - min_brightness))

            # Optional pause between breaths
            time.sleep(pause_time)

            # Breath in (increase brightness)
            for brightness in range(min_brightness, max_brightness + 1, breath_out_time+1):
                light.brightness = brightness
                time.sleep(breath_out_time / (max_brightness - min_brightness))

            # Optional pause between breaths
            time.sleep(pause_time)
            counter += 1
        light.brightness = max_brightness
    except KeyboardInterrupt:
        # Reset light when the script is interrupted
        light.brightness = max_brightness
        print("Breathing loop stopped.")


def setup_light_stim(bridge_ip='192.168.2.79', light_id="Dining table"):
    # Connect to the Philips Hue Bridge (replace 'your_bridge_ip' with your bridge's IP)
    bridge = Bridge(bridge_ip)

    # If you're connecting for the first time, press the button on the bridge and run the script
    bridge.connect()

    # Get the lights connected to the bridge
    lights = bridge.get_light_objects('name')  # Or 'id' if you want to use light IDs

    # Choose the light you want to control
    return lights[light_id]

#############
# EXAMPLE USAGE
#############
# if __name__ == "__main__":
#     # Set initial parameters for the calming breathing effect
#     min_brightness = 50  # The brightness for "breath out"
#     max_brightness = 254  # The brightness for "breath in" (max brightness in Hue API)
#     breath_in_time = 4  # Time (seconds) for breath in
#     breath_out_time = 6  # Time (seconds) for breath out
#     pause_time = 1
#     duration_light_stim = 2  # breathing cycles until stimulation stops
#
#     light = setup_light_stim()
#     pulsate_light(
#         light, breath_in_time, breath_out_time, min_brightness, max_brightness, pause_time, duration_light_stim
#         )
