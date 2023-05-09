#!/usr/bin/env python3 

import subprocess
import rospy

package = "main_pkg"
launch_file = "main_launch.launch"

class robots_spawner(object):
    def __init__(self, n_robots):
        self.n_robots = n_robots
    def spawning(self):
        rviz_on = int(input("Do you want to visualize the robots in RViz?\nChoose [0] if 'no'\nChoose [1] if 'yes'\nYour choice: "))
        if rviz_on == 0: rviz_command = "rviz_on:=false"
        elif rviz_on == 1: rviz_command = "rviz_on:=true"
        else: self.n_robots = -99999
        if self.n_robots == 0: print("Alright!")
        elif self.n_robots < 0: print("Impossible!")
        elif self.n_robots > 4: print("Impractical!")
        elif self.n_robots == 1:
            second_robot = "second_robot:=false"
            third_robot = "third_robot:=false"
            fourth_robot = "fourth_robot:=false"
            subprocess.Popen(["roslaunch", package, launch_file, rviz_command, second_robot, third_robot, fourth_robot])
        elif self.n_robots == 2:
            second_robot = "second_robot:=true"
            third_robot = "third_robot:=false"
            fourth_robot = "fourth_robot:=false"
            subprocess.Popen(["roslaunch", package, launch_file, rviz_command, second_robot, third_robot, fourth_robot])
        elif self.n_robots == 3:
            second_robot = "second_robot:=true"
            third_robot = "third_robot:=true"
            fourth_robot = "fourth_robot:=false"
            subprocess.Popen(["roslaunch", package, launch_file, rviz_command, second_robot, third_robot, fourth_robot])
        elif self.n_robots == 4:
            second_robot = "second_robot:=true"
            third_robot = "third_robot:=true"
            fourth_robot = "fourth_robot:=true"
            subprocess.Popen(["roslaunch", package, launch_file, rviz_command, second_robot, third_robot, fourth_robot])