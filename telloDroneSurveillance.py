"""
  :Description:
    Script for controlling a tello drone

  :Tutorials:
    - Simple code for starting the drone and basic setup
      https://www.youtube.com/watch?v=-Mb_FKhRn00&ab_channel=Murtaza%27sWorkshop-RoboticsandAI
"""

from tello import *

start()
power = get_battery()

print("Power level: ", power, "%")
takeoff()
land()

