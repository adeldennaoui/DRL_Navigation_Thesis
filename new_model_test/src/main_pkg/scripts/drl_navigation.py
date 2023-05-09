#!/usr/bin/env python3 

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker, MarkerArray
from squaternion import Quaternion

TIME_DELTA = 0.1
GOAL_REACHED = 0.8

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
    def __init__(self, env_dim):
        self.env_dim = env_dim
        self.last_odom = None
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.env_dim]]
        for m in range(self.env_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.env_dim]
            )
        self.gaps[-1][-1] += 0.03
        self.points_coordinates = np.zeros((20,2))
        rospy.init_node("navigation_node", anonymous=True)
        self.velodyne = rospy.Subscriber(
            "/point_cloud", PointCloud2, self.velodyne_cb, queue_size=1
        )
        self.odom = rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=1)
        self.goal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher1 = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
    
    def velodyne_cb(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.env_dim) * 5.5
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        self.points_coordinates[j] = np.array([data[i][0], data[i][1]])
                        break

    def odom_cb(self, msg):
        self.last_odom = msg

    def goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

    def state(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
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
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        #self.odom_z = self.last_odom.pose.pose.position.z
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
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
    
    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.02
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher1.publish(markerArray)
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

if __name__ == "__main__":
    try:
        file_name = "TD3"
        expl_noise = 1
        robot_dim = 4
        env_dim = 20
        goal_reached = False
        env = GazeboEnv(env_dim)
        time.sleep(10)
        state_dim = env_dim + robot_dim
        action_dim = 2
        max_action = 0.5
        network = TD3(state_dim, action_dim, max_action)
        network.load("/home/wsl-ros/new_model_test/src/main_pkg/scripts/pytorch_models", file_name)
        rospy.wait_for_message("/move_base_simple/goal", PoseStamped)
        print("A new goal is received!")
        gg_x = env.goal_x
        gg_y = env.goal_y
        state = env.state()
        while not goal_reached:
            env.goal_x = gg_x
            env.goal_y = gg_y
            done = False
            while not done:
                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in = [(action[0]+1)/2, 2*action[1]]
                next_state, done = env.step(a_in)
                state = next_state
            done = False
            rospy.wait_for_message("/move_base_simple/goal", PoseStamped)
            print("A new goal is received!")
            gg_x = env.goal_x
            gg_y = env.goal_y
            state = env.state()
    except rospy.ROSInterruptException:
        rospy.loginfo("Something's wrong, brother.")
