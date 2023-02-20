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

TIME_DELTA = 0.1
GOAL_REACHED = 0.5

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
        self.las_ran = 1.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.goal_x = 1.0
        self.goal_y = 0.0
        self.laser_data = np.ones(self.env_dim)
        self.last_odom = None
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        #self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
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
        self.laser = rospy.Subscriber("/r1/front_laser/scan", LaserScan, self.laser_cb, queue_size=1)
        self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_cb, queue_size=1)
        self.goal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)

    def laser_cb(self, msg):
        alfa = len(msg.ranges)/self.env_dim
        for i in range(self.env_dim):
            self.laser_data[i] = msg.ranges[round(((2*i+1)*alfa)/2)]
        self.las_ran = len(msg.ranges)

    def odom_cb(self, msg):
        self.last_odom = msg

    def goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

    def step(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)
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
        v_state = []
        v_state[:] = self.laser_data[:]
        laser_state = [v_state]
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if distance < GOAL_REACHED:
            done = True
            print('REACHED!')
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        return state, done
    
    #def reset(self, goal_):
    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state
        x = 0
        y = 0
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)
        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y
        self.publish_markers([0.0, 0.0])
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
        v_state = []
        v_state[:] = self.laser_data[:]
        laser_state = [v_state]
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state
    
    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
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
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
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

def obtain_waypoint(x_r, y_r, x_g, y_g, i, steps):
    waypoint = np.zeros(2)
    if i > steps:
        print("Shit's wrong")
        waypoint = np.array([x_r, y_r])
    else:
        waypoint[0] = x_r + (i/steps)*x_g
        waypoint[1] = y_r + (i/steps)*y_g
    return waypoint

"""
def extract_POI(dim, dist, laser_data):
    POIs = np.zeros(dim)
    for i in range(dim):
        if laser_data[i] > dist:
            POIs[i] = dist
    return POIs

def pick_waypoint(POIs, dim, dist, l1, l2, x_g, y_g, las_ran, x_r, y_r):
    scores = np.zeros(dim)
    alfa = math.pi / las_ran
    waypoint = np.zeros(2)
    POI_pos = np.zeros((dim,2))
    for i in range(dim):
        if POIs[i] != 0:
            angle = (2*i+1)/2*alfa
            diff_x = dist*math.cos(angle)
            diff_y = dist*math.sin(angle)
            if angle < math.pi/2:
                POI_pos[i][0]= x_r - diff_x
                POI_pos[i][1]= y_r + diff_y
            else:
                POI_pos[i][0]= x_r + diff_x
                POI_pos[i][1]= y_r + diff_y
            dist_c_r = np.linalg.norm([x_r-POI_pos[i][0], y_r-POI_pos[i][1]])
            up_t = math.exp(math.pow(dist_c_r/(l2-l1),2))
            down_t = math.exp(math.pow(l2/(l2-l1),2))
            term1 = math.tanh(up_t/down_t)*l2
            term2 = np.linalg.norm([x_g-POI_pos[i][0], y_g-POI_pos[i][1]])
            scores[i] = term1 + term2
        else:
            scores[i] = 0
    min_score = np.min(scores[np.nonzero(scores)])
    for i in range(dim):
        if scores[i] == min_score:
            waypoint_idx = i
    waypoint[0] = POI_pos[waypoint_idx][0]
    waypoint[1] = POI_pos[waypoint_idx][1]
    return waypoint
"""

if __name__ == '__main__':
    try:
        #seed = 0
        #l1 = 5
        #l2 = 10
        dist = 1
        th_goal = 0.5
        th_wp = 1
        expl_min = 0.1
        expl_decay_steps = 500000
        expl_noise = 1
        random_near_obtsacle = True
        count_rand_actions = 0
        env_dim = 20
        robot_dim = 4
        steps = 4
        launch_file = "main_launch.launch"
        file_name = "TD3"
        env = GazeboEnv(launch_file, env_dim)
        time.sleep(10)
        state_dim = env_dim + robot_dim
        action_dim = 2
        max_action = 0.5
        network = TD3(state_dim, action_dim, max_action)
        network.load("./pytorch_models", file_name)
        state = env.reset()
        goal_reached = False
        rospy.wait_for_message("/move_base_simple/goal", PoseStamped)
        gg_x = env.goal_x
        gg_y = env.goal_y
        """
        while goal_reached == False:
            if np.linalg.norm([env.odom_x - gg_x, env.odom_y - gg_y]) > th_goal:
                if np.linalg.norm([env.odom_x - gg_x, env.odom_y - gg_y]) > th_wp:
                    waypoint = pick_waypoint(
                        extract_POI(env_dim, dist, env.laser_data),
                        env_dim,
                        dist,
                        l1,
                        l2,
                        gg_x,
                        gg_y,
                        env.las_ran,
                        env.odom_x,
                        env.odom_y
                    )
                else:
                    waypoint = np.array([gg_x, gg_y])
                print(waypoint)
                env.goal_x = waypoint[0]
                env.goal_y = waypoint[1]
                done = False
                while not done:
                    action = network.get_action(np.array(state))
                    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                        -max_action, max_action
                    )
                    if random_near_obtsacle:
                        if (
                            np.random.uniform(0, 1) > 0.85
                            and min(state[4:-8]) < 0.6
                            and count_rand_actions < 1
                        ):
                            count_rand_actions = np.random.randint(8, 15)
                            random_action = np.random.uniform(-1, 1, 2)
                        if count_rand_actions > 0:
                            count_rand_actions -= 1
                            action = random_action
                            action[0] = -1
                    a_in = [(action[0]+1)/2, action[1]]
                    next_state, done = env.step(a_in)
                    state = next_state
                rospy.sleep(0.5)
                env.goal_x
            else:
                goal_reached = True
        """
        while not goal_reached:
            for i in range(steps):
                waypoint = obtain_waypoint(env.odom_x, env.odom_y, gg_x, gg_y, i, steps)
                print(waypoint)
                env.goal_x = waypoint[0]
                env.goal_y = waypoint[1]
                done = False
                while not done:
                    action = network.get_action(np.array(state))
                    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                        -max_action, max_action
                    )
                    if random_near_obtsacle:
                        if (
                            np.random.uniform(0, 1) > 0.85
                            and min(state[4:-8]) < 0.6
                            and count_rand_actions < 1
                        ):
                            count_rand_actions = np.random.randint(8, 15)
                            random_action = np.random.uniform(-1, 1, 2)
                        if count_rand_actions > 0:
                            count_rand_actions -= 1
                            action = random_action
                            action[0] = -1
                    a_in = [(action[0]+1)/2, 2*action[1]]
                    next_state, done = env.step(a_in)
                    state = next_state
                rospy.sleep(0.5)
                if i == steps - 1:
                    goal_reached = True
        rospy.signal_shutdown("We did it, bro!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Something's wrong, bro!")