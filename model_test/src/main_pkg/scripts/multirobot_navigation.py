#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from squaternion import Quaternion
import time
import subprocess
import os
from os import path
import random

TIME_DELTA = 0.1
GOAL_REACHED = 0.5

def check_pos(x, y):
    goal_ok = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False
    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False
    return goal_ok

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()
    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(device)
        self.max_action = max_action
    def get_action(self, state):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    def load(self, directory, filename):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )

class GazeboEnv(object):
    def __init__(self, launchfile, env_dim):
        self.env_dim = env_dim
        self.lower = -5.0
        self.upper = 5.0
        self.odom1_x = 0
        self.odom1_y = 0
        self.goal1_x = 1
        self.goal1_y = 0.0
        self.odom2_x = 0
        self.odom2_y = 0
        self.goal2_x = 1
        self.goal2_y = 0.0
        self.laser_data1 = np.ones(self.env_dim)
        self.laser_data2 = np.ones(self.env_dim)
        self.last_odom1 = None
        self.last_odom2 = None
        self.set_self_state1 = ModelState()
        self.set_self_state1.model_name = "r1"
        self.set_self_state1.pose.position.x = 0.0
        self.set_self_state1.pose.position.y = 0.0
        self.set_self_state1.pose.position.z = 0.0
        self.set_self_state1.pose.orientation.x = 0.0
        self.set_self_state1.pose.orientation.y = 0.0
        self.set_self_state1.pose.orientation.z = 0.0
        self.set_self_state1.pose.orientation.w = 1.0
        self.set_self_state2 = ModelState()
        self.set_self_state2.model_name = "r2"
        self.set_self_state2.pose.position.x = 0.8
        self.set_self_state2.pose.position.y = 0.8
        self.set_self_state2.pose.position.z = 0.0
        self.set_self_state2.pose.orientation.x = 0.0
        self.set_self_state2.pose.orientation.y = 0.0
        self.set_self_state2.pose.orientation.z = 0.0
        self.set_self_state2.pose.orientation.w = 1.0
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("roscore launched!")
        rospy.init_node("navigation_test", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "/home/student/model_test/src/main_pkg/launch", launchfile)
        if not path.exists(fullpath):
            os.system("killall -9 roscore rosmaster")
            raise IOError("File " + fullpath + " does not exist")
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo's working, baby!")
        self.vel_pub1 = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state1 = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.vel_pub2 = rospy.Publisher("/r2/cmd_vel", Twist, queue_size=1)
        self.set_state2 = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher0 = rospy.Publisher("goal_point1", MarkerArray, queue_size=3)
        self.publisher1 = rospy.Publisher("goal_point2", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity1", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("linear_velocity2", MarkerArray, queue_size=1)
        self.publisher4 = rospy.Publisher("angular_velocity1", MarkerArray, queue_size=1)
        self.publisher5 = rospy.Publisher("angular_velocity2", MarkerArray, queue_size=1)
        self.odom1 = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback1, queue_size=1
        )
        self.laser1 = rospy.Subscriber("/r1/front_laser/scan", LaserScan, self.laser_cb1, queue_size=1)
        self.laser2 = rospy.Subscriber("/r2/front_laser/scan", LaserScan, self.laser_cb2, queue_size=1)
        self.odom2 = rospy.Subscriber(
            "/r2/odom", Odometry, self.odom_callback2, queue_size=1
        )

    def laser_cb1(self, msg):
        alfa = len(msg.ranges)/self.env_dim
        for i in range(self.env_dim):
            self.laser_data1[i] = msg.ranges[round(((2*i+1)*alfa)/2)]
        self.las_ran1 = len(msg.ranges)
    
    def laser_cb2(self, msg):
        alfa = len(msg.ranges)/self.env_dim
        for i in range(self.env_dim):
            self.laser_data2[i] = msg.ranges[round(((2*i+1)*alfa)/2)]
        self.las_ran2 = len(msg.ranges)

    def odom_callback1(self, od_data):
        self.last_odom1 = od_data

    def odom_callback2(self, od_data):
        self.last_odom2 = od_data

    def step(self, action1, action2):
        done1 = False
        done2 = False
        vel_cmd1 = Twist()
        vel_cmd2 = Twist()
        vel_cmd1.linear.x = action1[0]
        vel_cmd1.angular.z = action1[1]
        vel_cmd2.linear.x = action2[0]
        vel_cmd2.angular.z = action2[1]
        self.vel_pub1.publish(vel_cmd1)
        self.vel_pub2.publish(vel_cmd2)
        self.publish_markers(action1, action2)
        done = False
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed - STEP")
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state1 = []
        v_state1[:] = self.laser_data1[:]
        laser_state1 = [v_state1]
        v_state2 = []
        v_state2[:] = self.laser_data2[:]
        laser_state2 = [v_state2]
        self.odom1_x = self.last_odom1.pose.pose.position.x
        self.odom1_y = self.last_odom1.pose.pose.position.y
        self.odom2_x = self.last_odom2.pose.pose.position.x
        self.odom2_y = self.last_odom2.pose.pose.position.y
        quaternion1 = Quaternion(
            self.last_odom1.pose.pose.orientation.w,
            self.last_odom1.pose.pose.orientation.x,
            self.last_odom1.pose.pose.orientation.y,
            self.last_odom1.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        euler1 = quaternion2.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        angle1 = round(euler1[2], 4)
        angle2 = round(euler2[2], 4)
        distance1 = np.linalg.norm(
            [self.odom1_x - self.goal1_x, self.odom1_y - self.goal1_y]
        )
        distance2 = np.linalg.norm(
            [self.odom2_x - self.goal2_x, self.odom2_y - self.goal2_y]
        )
        skew_x1 = self.goal1_x - self.odom1_x
        skew_y1 = self.goal1_y - self.odom1_y
        skew_x2 = self.goal2_x - self.odom2_x
        skew_y2 = self.goal2_y - self.odom2_y
        dot1 = skew_x1 * 1 + skew_y1 * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        mag11 = math.sqrt(math.pow(skew_x1, 2) + math.pow(skew_y1, 2))
        mag21 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta1 = math.acos(dot1 / (mag11 * mag21))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        if skew_y1 < 0:
            if skew_x1 < 0:
                beta1 = -beta1
            else:
                beta1 = 0 - beta1
        theta1 = beta1 - angle1
        if theta1 > np.pi:
            theta1 = np.pi - theta1
            theta1 = -np.pi - theta1
        if theta1 < -np.pi:
            theta1 = -np.pi - theta1
            theta1 = np.pi - theta1
        if distance1 < GOAL_REACHED:
            done1 = True
            print('R1 REACHED!')
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        theta2 = beta2 - angle2
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        if distance2 < GOAL_REACHED:
            done2 = True
            print('R2 REACHED!')
        robot_state1 = [distance1, theta1, action1[0], action1[1]]
        robot_state2 = [distance2, theta2, action2[0], action2[1]]
        state1 = np.append(laser_state1, robot_state1)
        state2 = np.append(laser_state2, robot_state2)
        return state1, done1, state2, done2
    
    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        angle1 = np.random.uniform(-np.pi, np.pi)
        quaternion1 = Quaternion.from_euler(0.0, 0.0, angle1)
        angle2 = np.random.uniform(-np.pi, np.pi)
        quaternion2 = Quaternion.from_euler(0.0, 0.0, angle2)
        object_state1 = self.set_self_state1
        object_state2 = self.set_self_state2
        x1 = 0
        y1 = 0
        x2 = 0.8
        y2 = 0.8
        object_state1.pose.position.x = x1
        object_state1.pose.position.y = y1
        object_state1.pose.orientation.x = quaternion1.x
        object_state1.pose.orientation.y = quaternion1.y
        object_state1.pose.orientation.z = quaternion1.z
        object_state1.pose.orientation.w = quaternion1.w
        object_state2.pose.position.x = x2
        object_state2.pose.position.y = y2
        object_state2.pose.orientation.x = quaternion2.x
        object_state2.pose.orientation.y = quaternion2.y
        object_state2.pose.orientation.z = quaternion2.z
        object_state2.pose.orientation.w = quaternion2.w
        self.set_state1.publish(object_state1)
        self.set_state2.publish(object_state2)
        self.odom1_x = object_state1.pose.position.x
        self.odom1_y = object_state1.pose.position.y
        self.odom2_x = object_state2.pose.position.x
        self.odom2_y = object_state2.pose.position.y
        self.change_goal1()
        self.change_goal2()
        self.publish_markers([0.0, 0.0], [0.0, 0.0])
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed - RESET")
        time.sleep(TIME_DELTA)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state1 = []
        v_state1[:] = self.laser_data1[:]
        laser_state1 = [v_state1]
        v_state2 = []
        v_state2[:] = self.laser_data2[:]
        laser_state2 = [v_state2]
        distance1 = np.linalg.norm(
            [self.odom1_x - self.goal1_x, self.odom1_y - self.goal1_y]
        )
        distance2 = np.linalg.norm(
            [self.odom2_x - self.goal2_x, self.odom2_y - self.goal2_y]
        )
        skew_x1 = self.goal1_x - self.odom1_x
        skew_y1 = self.goal1_y - self.odom1_y
        skew_x2 = self.goal2_x - self.odom2_x
        skew_y2 = self.goal2_y - self.odom2_y
        dot1 = skew_x1 * 1 + skew_y1 * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        mag11 = math.sqrt(math.pow(skew_x1, 2) + math.pow(skew_y1, 2))
        mag21 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta1 = math.acos(dot1 / (mag11 * mag21))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        if skew_y1 < 0:
            if skew_x1 < 0:
                beta1 = -beta1
            else:
                beta1 = 0 - beta1
        theta1 = beta1 - angle1
        if theta1 > np.pi:
            theta1 = np.pi - theta1
            theta1 = -np.pi - theta1
        if theta1 < -np.pi:
            theta1 = -np.pi - theta1
            theta1 = np.pi - theta1
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        theta2 = beta2 - angle2
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        robot_state1 = [distance1, theta1, 0.0, 0.0]
        robot_state2 = [distance2, theta2, 0.0, 0.0]
        state1 = np.append(laser_state1, robot_state1)
        state2 = np.append(laser_state2, robot_state2)
        return state1, state2

    def publish_markers(self, action1, action2):
        markerArray0 = MarkerArray()
        marker0 = Marker()
        marker0.header.frame_id = "odom"
        marker0.type = marker0.CYLINDER
        marker0.action = marker0.ADD
        marker0.scale.x = 0.1
        marker0.scale.y = 0.1
        marker0.scale.z = 0.01
        marker0.color.a = 1.0
        marker0.color.r = 0.0
        marker0.color.g = 1.0
        marker0.color.b = 0.0
        marker0.pose.orientation.w = 1.0
        marker0.pose.position.x = self.goal1_x
        marker0.pose.position.y = self.goal1_y
        marker0.pose.position.z = 0
        markerArray0.markers.append(marker0)
        markerArray1 = MarkerArray()
        marker1 = Marker()
        marker1.header.frame_id = "odom"
        marker1.type = marker1.CYLINDER
        marker1.action = marker1.ADD
        marker1.scale.x = 0.1
        marker1.scale.y = 0.1
        marker1.scale.z = 0.01
        marker1.color.a = 1.0
        marker1.color.r = 0.0
        marker1.color.g = 1.0
        marker1.color.b = 0.0
        marker1.pose.orientation.w = 1.0
        marker1.pose.position.x = self.goal2_x
        marker1.pose.position.y = self.goal2_y
        marker1.pose.position.z = 0
        markerArray1.markers.append(marker1)
        self.publisher0.publish(markerArray0)
        self.publisher1.publish(markerArray1)
        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker0.CUBE
        marker2.action = marker0.ADD
        marker2.scale.x = abs(action1[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0
        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)
        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker0.CUBE
        marker3.action = marker0.ADD
        marker3.scale.x = abs(action1[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0
        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)
        markerArray4 = MarkerArray()
        marker4 = Marker()
        marker4.header.frame_id = "odom"
        marker4.type = marker1.CUBE
        marker4.action = marker1.ADD
        marker4.scale.x = abs(action2[0])
        marker4.scale.y = 0.1
        marker4.scale.z = 0.01
        marker4.color.a = 1.0
        marker4.color.r = 1.0
        marker4.color.g = 0.0
        marker4.color.b = 0.0
        marker4.pose.orientation.w = 1.0
        marker4.pose.position.x = 5
        marker4.pose.position.y = 0
        marker4.pose.position.z = 0
        markerArray4.markers.append(marker4)
        self.publisher4.publish(markerArray4)
        markerArray5 = MarkerArray()
        marker5 = Marker()
        marker5.header.frame_id = "odom"
        marker5.type = marker1.CUBE
        marker5.action = marker1.ADD
        marker5.scale.x = abs(action2[1])
        marker5.scale.y = 0.1
        marker5.scale.z = 0.01
        marker5.color.a = 1.0
        marker5.color.r = 1.0
        marker5.color.g = 0.0
        marker5.color.b = 0.0
        marker5.pose.orientation.w = 1.0
        marker5.pose.position.x = 5
        marker5.pose.position.y = 0.2
        marker5.pose.position.z = 0
        markerArray5.markers.append(marker5)
        self.publisher5.publish(markerArray5)

    def change_goal1(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004
        goal_ok1 = False
        while not goal_ok1:
            self.goal1_x = self.odom1_x + random.uniform(self.upper, self.lower)
            self.goal1_y = self.odom1_y + random.uniform(self.upper, self.lower)
            goal_ok1 = check_pos(self.goal1_x, self.goal1_y)

    def change_goal2(self):
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004
        goal_ok2 = False
        while not goal_ok2:
            self.goal2_x = self.odom2_x + random.uniform(self.upper, self.lower)
            self.goal2_y = self.odom2_y + random.uniform(self.upper, self.lower)
            goal_ok2 = check_pos(self.goal1_x, self.goal1_y)

if __name__ == "__main__":
    try:
        seed = 0
        count1 = 0
        count2 = 0
        done1 = False
        done2 = False
        expl_min = 0.1
        expl_decay_steps = 500000
        expl_noise = 1
        random_near_obtsacle = True
        count_rand_actions1 = 0
        count_rand_actions2 = 0
        env_dim = 20
        robot_dim = 4
        steps = 4
        launch_file = "main_launch1.launch"
        file_name = "TD3"
        env = GazeboEnv(launch_file, env_dim)
        time.sleep(10)
        state_dim = env_dim + robot_dim
        action_dim = 2
        max_action = 1
        network = TD3(state_dim, action_dim, max_action)
        network.load("./pytorch_models", file_name)
        state1, state2 = env.reset()
        print("goal for robot1: ", env.goal1_x, " and ", env.goal1_y)
        print("goal for robot1: ", env.goal2_x, " and ", env.goal2_y)
        time.sleep(10)
        if expl_noise > expl_min:
                expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
        while done1==False and done2==False:
            action1 = network.get_action(np.array(state1))
            action2 = network.get_action(np.array(state2))
            action1 = (action1 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                -max_action, max_action
            )
            action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                -max_action, max_action
            )
            if random_near_obtsacle:
                if (
                np.random.uniform(0, 1) > 0.85
                and min(state1[4:-8]) < 0.6
                and count_rand_actions1 < 1
                ):
                    count_rand_actions1 = np.random.randint(8, 15)
                    random_action1 = np.random.uniform(-1, 1, 2)
                if count_rand_actions1 > 0:
                    count_rand_actions1 -= 1
                    action1 = random_action1
                    action1[0] = -1
                if (
                np.random.uniform(0, 1) > 0.85
                and min(state2[4:-8]) < 0.6
                and count_rand_actions2 < 1
                ):
                    count_rand_actions2 = np.random.randint(8, 15)
                    random_action2 = np.random.uniform(-1, 1, 2)
                if count_rand_actions2 > 0:
                    count_rand_actions2 -= 1
                    action2 = random_action2
                    action2[0] = -1
            a_in1 = [(action1[0]+1)/2, 2*action1[1]]
        #time.sleep(0.1)
            a_in2 = [(action2[0]+1)/2, 2*action2[1]]
        #time.sleep(0.1)
            next_state1, done1, next_state2, done2 = env.step(a_in1, a_in2)
            state1 = next_state1
            state2 = next_state2
            if done1 == True:
                env.change_goal1()
                env.publish_markers(action1, action2)
                print("goal for robot1: ", env.goal1_x, " and ", env.goal1_y)
                done1 = False
                count1 += 1
            if done2 == True:
                env.change_goal2()
                env.publish_markers(action1, action2)
                print("goal for robot1: ", env.goal2_x, " and ", env.goal2_y)
                done2 = False
                count2 += 1
            if count1 > 5:
                done1 = True
                print("R1 stops!")
            if count1 > 5:
                done2 = True
                print("R1 stops!")
        rospy.sleep(0.5)
        rospy.signal_shutdown("IT WORKED, DICKHEAD!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Something's wrong bro!")