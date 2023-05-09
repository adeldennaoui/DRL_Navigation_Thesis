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
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Empty
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
    def __init__(self, n_robots, env_dim):
        self.env_dim = env_dim
        self.n_robots = n_robots
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.env_dim]]
        for m in range(self.env_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.env_dim]
            )
        self.gaps[-1][-1] += 0.03
        self.gaps2 = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.env_dim]]
        for m in range(self.env_dim - 1):
            self.gaps2.append(
                [self.gaps2[m][1], self.gaps2[m][1] + np.pi / self.env_dim]
            )
        self.gaps2[-1][-1] += 0.03
        self.gaps3 = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.env_dim]]
        for m in range(self.env_dim - 1):
            self.gaps3.append(
                [self.gaps3[m][1], self.gaps3[m][1] + np.pi / self.env_dim]
            )
        self.gaps3[-1][-1] += 0.03
        self.gaps4 = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.env_dim]]
        for m in range(self.env_dim - 1):
            self.gaps4.append(
                [self.gaps4[m][1], self.gaps4[m][1] + np.pi / self.env_dim]
            )
        self.gaps4[-1][-1] += 0.03
        rospy.init_node("navigation_node", anonymous=False)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher1 = rospy.Publisher("goal_point1", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("goal_point2", MarkerArray, queue_size=3)
        self.publisher3 = rospy.Publisher("goal_point3", MarkerArray, queue_size=3)
        self.publisher4 = rospy.Publisher("goal_point4", MarkerArray, queue_size=3)        
        self.points_coordinates = np.zeros((20,2))
        self.points_coordinates2 = np.zeros((20,2))
        self.points_coordinates3 = np.zeros((20,2))
        self.points_coordinates4 = np.zeros((20,2))
        if n_robots == 1:
            self.last_odom = None
            self.velodyne = rospy.Subscriber(
                "/velodyne_points1", PointCloud2, self.velodyne_cb, queue_size=1
            )
            self.goal = rospy.Subscriber("/goal", PoseStamped, self.goal_cb, queue_size=10)
            self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
            self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_cb, queue_size=1)
        elif n_robots == 2:
            self.last_odom = None
            self.last_odom2 = None
            self.velodyne = rospy.Subscriber(
                "/velodyne_points1", PointCloud2, self.velodyne_cb, queue_size=1
            )
            self.velodyne2 = rospy.Subscriber(
                "/velodyne_points2", PointCloud2, self.velodyne2_cb, queue_size=1
            )
            self.goal = rospy.Subscriber("/goal", PoseStamped, self.goal_cb, queue_size=10)
            self.goal2 = rospy.Subscriber("/goal2", PoseStamped, self.goal2_cb, queue_size=10)
            self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
            self.vel_pub2 = rospy.Publisher("/r2/cmd_vel", Twist, queue_size=1)
            self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_cb, queue_size=1)
            self.odom2 = rospy.Subscriber("/r2/odom", Odometry, self.odom2_cb, queue_size=1)
        elif n_robots == 3:
            self.last_odom = None
            self.last_odom2 = None
            self.last_odom3 = None
            self.velodyne = rospy.Subscriber(
                "/velodyne_points1", PointCloud2, self.velodyne_cb, queue_size=1
            )
            self.velodyne2 = rospy.Subscriber(
                "/velodyne_points2", PointCloud2, self.velodyne2_cb, queue_size=1
            )
            self.velodyne3 = rospy.Subscriber(
                "/velodyne_points3", PointCloud2, self.velodyne3_cb, queue_size=1
            )
            self.goal = rospy.Subscriber("/goal", PoseStamped, self.goal_cb, queue_size=10)
            self.goal2 = rospy.Subscriber("/goal2", PoseStamped, self.goal2_cb, queue_size=10)
            self.goal3 = rospy.Subscriber("/goal3", PoseStamped, self.goal3_cb, queue_size=10)          
            self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
            self.vel_pub2 = rospy.Publisher("/r2/cmd_vel", Twist, queue_size=1)
            self.vel_pub3 = rospy.Publisher("/r3/cmd_vel", Twist, queue_size=1)
            self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_cb, queue_size=1)
            self.odom2 = rospy.Subscriber("/r2/odom", Odometry, self.odom2_cb, queue_size=1)
            self.odom3 = rospy.Subscriber("/r3/odom", Odometry, self.odom3_cb, queue_size=1)
        elif n_robots == 4:
            self.last_odom = None
            self.last_odom2 = None
            self.last_odom3 = None
            self.last_odom4 = None
            self.velodyne = rospy.Subscriber(
                "/velodyne_points1", PointCloud2, self.velodyne_cb, queue_size=1
            )
            self.velodyne2 = rospy.Subscriber(
                "/velodyne_points2", PointCloud2, self.velodyne2_cb, queue_size=1
            )
            self.velodyne3 = rospy.Subscriber(
                "/velodyne_points3", PointCloud2, self.velodyne3_cb, queue_size=1
            )
            self.velodyne4 = rospy.Subscriber(
                "/velodyne_points4", PointCloud2, self.velodyne4_cb, queue_size=1
            )
            self.goal = rospy.Subscriber("/goal", PoseStamped, self.goal_cb, queue_size=10)
            self.goal2 = rospy.Subscriber("/goal2", PoseStamped, self.goal2_cb, queue_size=10)
            self.goal3 = rospy.Subscriber("/goal3", PoseStamped, self.goal3_cb, queue_size=10)    
            self.goal4 = rospy.Subscriber("/goal4", PoseStamped, self.goal4_cb, queue_size=10)       
            self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
            self.vel_pub2 = rospy.Publisher("/r2/cmd_vel", Twist, queue_size=1)
            self.vel_pub3 = rospy.Publisher("/r3/cmd_vel", Twist, queue_size=1)
            self.vel_pub4 = rospy.Publisher("/r4/cmd_vel", Twist, queue_size=1)
            self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_cb, queue_size=1)
            self.odom2 = rospy.Subscriber("/r2/odom", Odometry, self.odom2_cb, queue_size=1)
            self.odom3 = rospy.Subscriber("/r3/odom", Odometry, self.odom3_cb, queue_size=1)
            self.odom4 = rospy.Subscriber("/r4/odom", Odometry, self.odom4_cb, queue_size=1)

    def odom_cb(self, msg):
        self.last_odom = msg

    def odom2_cb(self, msg):
        self.last_odom2 = msg

    def odom3_cb(self, msg):
        self.last_odom3 = msg

    def odom4_cb(self, msg):
        self.last_odom4 = msg

    def goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

    def goal2_cb(self, msg):
        self.goal_x2 = msg.pose.position.x
        self.goal_y2 = msg.pose.position.y

    def goal3_cb(self, msg):
        self.goal_x3 = msg.pose.position.x
        self.goal_y3 = msg.pose.position.y

    def goal4_cb(self, msg):
        self.goal_x4 = msg.pose.position.x
        self.goal_y4 = msg.pose.position.y

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

    def velodyne2_cb(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data2 = np.ones(self.env_dim) * 5.5
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps2[j][0] <= beta < self.gaps2[j][1]:
                        self.velodyne_data2[j] = min(self.velodyne_data2[j], dist)
                        self.points_coordinates2[j] = np.array([data[i][0], data[i][1]])
                        break

    def velodyne3_cb(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data3 = np.ones(self.env_dim) * 5.5
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps3[j][0] <= beta < self.gaps3[j][1]:
                        self.velodyne_data3[j] = min(self.velodyne_data3[j], dist)
                        self.points_coordinates3[j] = np.array([data[i][0], data[i][1]])
                        break
    def velodyne4_cb(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data4 = np.ones(self.env_dim) * 5.5
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                for j in range(len(self.gaps)):
                    if self.gaps4[j][0] <= beta < self.gaps4[j][1]:
                        self.velodyne_data4[j] = min(self.velodyne_data4[j], dist)
                        self.points_coordinates4[j] = np.array([data[i][0], data[i][1]])
                        break

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
    
    def state2(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_x2 = self.last_odom2.pose.pose.position.x
        odom_y2 = self.last_odom2.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        angle = round(euler[2], 4)
        angle2 = round(euler2[2], 4)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        laser_state = [v_state]
        laser_state2 = [v_state2]
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        distance2 = np.linalg.norm(
            [odom_x2 - self.goal_x2, odom_y2 - self.goal_y2]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
        skew_x2 = self.goal_x2 - odom_x2
        skew_y2 = self.goal_y2 - odom_y2
        dot = skew_x * 1 + skew_y * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        theta = beta - angle
        theta2 = beta2 - angle2
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        robot_state = [distance, theta, 0.0, 0.0]
        robot_state2 = [distance2, theta2, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        state2 = np.append(laser_state2, robot_state2)
        return state, state2

    def state3(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_x2 = self.last_odom2.pose.pose.position.x
        odom_y2 = self.last_odom2.pose.pose.position.y
        odom_x3 = self.last_odom3.pose.pose.position.x
        odom_y3 = self.last_odom3.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        quaternion3 = Quaternion(
            self.last_odom3.pose.pose.orientation.w,
            self.last_odom3.pose.pose.orientation.x,
            self.last_odom3.pose.pose.orientation.y,
            self.last_odom3.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        euler3 = quaternion3.to_euler(degrees=False)
        angle = round(euler[2], 4)
        angle2 = round(euler2[2], 4)
        angle3 = round(euler3[2], 4)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        v_state3 = []
        v_state3[:] = self.velodyne_data3[:]
        laser_state = [v_state]
        laser_state2 = [v_state2]
        laser_state3 = [v_state3]
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        distance2 = np.linalg.norm(
            [odom_x2 - self.goal_x2, odom_y2 - self.goal_y2]
        )
        distance3 = np.linalg.norm(
            [odom_x3 - self.goal_x3, odom_y3 - self.goal_y3]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
        skew_x2 = self.goal_x2 - odom_x2
        skew_y2 = self.goal_y2 - odom_y2
        skew_x3 = self.goal_x3 - odom_x3
        skew_y3 = self.goal_y3 - odom_y3
        dot = skew_x * 1 + skew_y * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        dot3 = skew_x3 * 1 + skew_y3 * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag13 = math.sqrt(math.pow(skew_x3, 2) + math.pow(skew_y3, 2))
        mag23 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        beta3 = math.acos(dot3 / (mag13 * mag23))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        if skew_y3 < 0:
            if skew_x3 < 0:
                beta3 = -beta3
            else:
                beta3 = 0 - beta3
        theta = beta - angle
        theta2 = beta2 - angle2
        theta3 = beta3 - angle3
        if skew_y3 < 0:
            if skew_x3 < 0:
                beta3 = -beta3
            else:
                beta3 = 0 - beta3
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        if theta3 > np.pi:
            theta3 = np.pi - theta3
            theta3 = -np.pi - theta3
        if theta3 < -np.pi:
            theta3 = -np.pi - theta3
            theta3 = np.pi - theta3
        robot_state = [distance, theta, 0.0, 0.0]
        robot_state2 = [distance2, theta2, 0.0, 0.0]
        robot_state3 = [distance3, theta3, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        state2 = np.append(laser_state2, robot_state2)
        state3 = np.append(laser_state3, robot_state3)
        return state, state2, state3

    def state4(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_x2 = self.last_odom2.pose.pose.position.x
        odom_y2 = self.last_odom2.pose.pose.position.y
        odom_x3 = self.last_odom3.pose.pose.position.x
        odom_y3 = self.last_odom3.pose.pose.position.y
        odom_x4 = self.last_odom4.pose.pose.position.x
        odom_y4 = self.last_odom4.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        quaternion3 = Quaternion(
            self.last_odom3.pose.pose.orientation.w,
            self.last_odom3.pose.pose.orientation.x,
            self.last_odom3.pose.pose.orientation.y,
            self.last_odom3.pose.pose.orientation.z,
        )
        quaternion4 = Quaternion(
            self.last_odom4.pose.pose.orientation.w,
            self.last_odom4.pose.pose.orientation.x,
            self.last_odom4.pose.pose.orientation.y,
            self.last_odom4.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        euler3 = quaternion3.to_euler(degrees=False)
        euler4 = quaternion4.to_euler(degrees=False)
        angle = round(euler[2], 4)
        angle2 = round(euler2[2], 4)
        angle3 = round(euler3[2], 4)
        angle4 = round(euler4[2], 4)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        v_state3 = []
        v_state3[:] = self.velodyne_data3[:]
        v_state4 = []
        v_state4[:] = self.velodyne_data4[:]
        laser_state = [v_state]
        laser_state2 = [v_state2]
        laser_state3 = [v_state3]
        laser_state4 = [v_state4]
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        distance2 = np.linalg.norm(
            [odom_x2 - self.goal_x2, odom_y2 - self.goal_y2]
        )
        distance3 = np.linalg.norm(
            [odom_x3 - self.goal_x3, odom_y3 - self.goal_y3]
        )
        distance4 = np.linalg.norm(
            [odom_x4 - self.goal_x4, odom_y4 - self.goal_y4]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
        skew_x2 = self.goal_x2 - odom_x2
        skew_y2 = self.goal_y2 - odom_y2
        skew_x3 = self.goal_x3 - odom_x3
        skew_y3 = self.goal_y3 - odom_y3
        skew_x4 = self.goal_x4 - odom_x4
        skew_y4 = self.goal_y4 - odom_y4
        dot = skew_x * 1 + skew_y * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        dot3 = skew_x3 * 1 + skew_y3 * 0
        dot4 = skew_x4 * 1 + skew_y4 * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag13 = math.sqrt(math.pow(skew_x3, 2) + math.pow(skew_y3, 2))
        mag23 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag14 = math.sqrt(math.pow(skew_x4, 2) + math.pow(skew_y4, 2))
        mag24 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        beta3 = math.acos(dot3 / (mag13 * mag23))
        beta4 = math.acos(dot4 / (mag14 * mag24))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        if skew_y3 < 0:
            if skew_x3 < 0:
                beta3 = -beta3
            else:
                beta3 = 0 - beta3
        if skew_y4 < 0:
            if skew_x4 < 0:
                beta4 = -beta4
            else:
                beta4 = 0 - beta4
        theta = beta - angle
        theta2 = beta2 - angle2
        theta3 = beta3 - angle3
        theta4 = beta4 - angle4
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        if theta3 > np.pi:
            theta3 = np.pi - theta3
            theta3 = -np.pi - theta3
        if theta3 < -np.pi:
            theta3 = -np.pi - theta3
            theta3 = np.pi - theta3
        if theta > np.pi:
            theta4 = np.pi - theta4
            theta4 = -np.pi - theta4
        if theta4 < -np.pi:
            theta4 = -np.pi - theta4
            theta4 = np.pi - theta4
        robot_state = [distance, theta, 0.0, 0.0]
        robot_state2 = [distance2, theta2, 0.0, 0.0]
        robot_state3 = [distance3, theta3, 0.0, 0.0]
        robot_state4 = [distance4, theta4, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        state2 = np.append(laser_state2, robot_state2)
        state3 = np.append(laser_state3, robot_state3)
        state4 = np.append(laser_state4, robot_state4)
        return state, state2, state3, state4

    def step(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers1()
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
    
    def step2(self, action, action2):
        vel_cmd = Twist()
        vel_cmd2 = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        vel_cmd2.linear.x = action2[0]
        vel_cmd2.angular.z = action2[1]
        self.vel_pub.publish(vel_cmd)
        self.vel_pub2.publish(vel_cmd2)
        self.publish_markers1()
        self.publish_markers2()
        done = False
        done2 = False
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
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        laser_state2 = [v_state2]
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_x2 = self.last_odom2.pose.pose.position.x
        odom_y2 = self.last_odom2.pose.pose.position.y
        #self.odom_z = self.last_odom.pose.pose.position.z
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        angle = round(euler[2], 4)
        angle2 = round(euler2[2], 4)
        distance = np.linalg.norm(
                [odom_x - self.goal_x, odom_y - self.goal_y]
            )
        distance2 = np.linalg.norm(
                [odom_x2 - self.goal_x2, odom_y2 - self.goal_y2]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
        skew_x2 = self.goal_x2 - odom_x2
        skew_y2 = self.goal_y2 - odom_y2
        dot = skew_x * 1 + skew_y * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        theta = beta - angle
        theta2 = beta2 - angle2
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        if distance < GOAL_REACHED:
            done = True
            print('R1 REACHED!')
        if distance2 < GOAL_REACHED:
            done2 = True
            print('R2 REACHED!')
        robot_state = [distance, theta, action[0], action[1]]
        robot_state2 = [distance2, theta2, action2[0], action2[1]]
        state = np.append(laser_state, robot_state)
        state2 = np.append(laser_state2, robot_state2)
        return state, state2, done, done2
    
    def step3(self, action, action2, action3):
        vel_cmd = Twist()
        vel_cmd2 = Twist()
        vel_cmd3 = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        vel_cmd2.linear.x = action2[0]
        vel_cmd2.angular.z = action2[1]
        vel_cmd3.linear.x = action3[0]
        vel_cmd3.angular.z = action3[1]
        self.vel_pub.publish(vel_cmd)
        self.vel_pub2.publish(vel_cmd2)
        self.vel_pub3.publish(vel_cmd3)
        self.publish_markers1()
        self.publish_markers2()
        self.publish_markers3()
        done = False
        done2 = False
        done3 = False
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
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        laser_state2 = [v_state2]
        v_state3 = []
        v_state3[:] = self.velodyne_data3[:]
        laser_state3 = [v_state3]
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_x2 = self.last_odom2.pose.pose.position.x
        odom_y2 = self.last_odom2.pose.pose.position.y
        odom_x3 = self.last_odom3.pose.pose.position.x
        odom_y3 = self.last_odom3.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        quaternion3 = Quaternion(
            self.last_odom3.pose.pose.orientation.w,
            self.last_odom3.pose.pose.orientation.x,
            self.last_odom3.pose.pose.orientation.y,
            self.last_odom3.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        euler3 = quaternion3.to_euler(degrees=False)
        angle = round(euler[2], 4)
        angle2 = round(euler2[2], 4)
        angle3 = round(euler3[2], 4)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        v_state3 = []
        v_state3[:] = self.velodyne_data3[:]
        laser_state = [v_state]
        laser_state2 = [v_state2]
        laser_state3 = [v_state3]
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        distance2 = np.linalg.norm(
            [odom_x2 - self.goal_x2, odom_y2 - self.goal_y2]
        )
        distance3 = np.linalg.norm(
            [odom_x3 - self.goal_x3, odom_y3 - self.goal_y3]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
        skew_x2 = self.goal_x2 - odom_x2
        skew_y2 = self.goal_y2 - odom_y2
        skew_x3 = self.goal_x3 - odom_x3
        skew_y3 = self.goal_y3 - odom_y3
        dot = skew_x * 1 + skew_y * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        dot3 = skew_x3 * 1 + skew_y3 * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag13 = math.sqrt(math.pow(skew_x3, 2) + math.pow(skew_y3, 2))
        mag23 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        beta3 = math.acos(dot3 / (mag13 * mag23))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        if skew_y3 < 0:
            if skew_x3 < 0:
                beta3 = -beta3
            else:
                beta3 = 0 - beta3
        theta = beta - angle
        theta2 = beta2 - angle2
        theta3 = beta3 - angle3
        if skew_y3 < 0:
            if skew_x3 < 0:
                beta3 = -beta3
            else:
                beta3 = 0 - beta3
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        if theta3 > np.pi:
            theta3 = np.pi - theta3
            theta3 = -np.pi - theta3
        if theta3 < -np.pi:
            theta3 = -np.pi - theta3
            theta3 = np.pi - theta3
        if distance < GOAL_REACHED:
            done = True
            print('R1 REACHED!')
        if distance2 < GOAL_REACHED:
            done2 = True
            print('R2 REACHED!')
        if distance3 < GOAL_REACHED:
            done3 = True
            print('R3 REACHED!')
        robot_state = [distance, theta, 0.0, 0.0]
        robot_state2 = [distance2, theta2, 0.0, 0.0]
        robot_state3 = [distance3, theta3, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        state2 = np.append(laser_state2, robot_state2)
        state3 = np.append(laser_state3, robot_state3)
        return state, state2, state3, done, done2, done3
    
    def step4(self, action, action2, action3, action4):
        vel_cmd = Twist()
        vel_cmd2 = Twist()
        vel_cmd3 = Twist()
        vel_cmd4 = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        vel_cmd2.linear.x = action2[0]
        vel_cmd2.angular.z = action2[1]
        vel_cmd3.linear.x = action3[0]
        vel_cmd3.angular.z = action3[1]
        vel_cmd4.linear.x = action4[0]
        vel_cmd4.angular.z = action4[1]
        self.vel_pub.publish(vel_cmd)
        self.vel_pub2.publish(vel_cmd2)
        self.vel_pub3.publish(vel_cmd3)
        self.vel_pub4.publish(vel_cmd4)
        self.publish_markers1()
        self.publish_markers2()
        self.publish_markers3()
        self.publish_markers4()
        done = False
        done2 = False
        done3 = False
        done4 = False
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
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        laser_state2 = [v_state2]
        v_state3 = []
        v_state3[:] = self.velodyne_data3[:]
        laser_state3 = [v_state3]
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_x2 = self.last_odom2.pose.pose.position.x
        odom_y2 = self.last_odom2.pose.pose.position.y
        odom_x3 = self.last_odom3.pose.pose.position.x
        odom_y3 = self.last_odom3.pose.pose.position.y
        odom_x4 = self.last_odom4.pose.pose.position.x
        odom_y4 = self.last_odom4.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        quaternion2 = Quaternion(
            self.last_odom2.pose.pose.orientation.w,
            self.last_odom2.pose.pose.orientation.x,
            self.last_odom2.pose.pose.orientation.y,
            self.last_odom2.pose.pose.orientation.z,
        )
        quaternion3 = Quaternion(
            self.last_odom3.pose.pose.orientation.w,
            self.last_odom3.pose.pose.orientation.x,
            self.last_odom3.pose.pose.orientation.y,
            self.last_odom3.pose.pose.orientation.z,
        )
        quaternion4 = Quaternion(
            self.last_odom4.pose.pose.orientation.w,
            self.last_odom4.pose.pose.orientation.x,
            self.last_odom4.pose.pose.orientation.y,
            self.last_odom4.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        euler2 = quaternion2.to_euler(degrees=False)
        euler3 = quaternion3.to_euler(degrees=False)
        euler4 = quaternion4.to_euler(degrees=False)
        angle = round(euler[2], 4)
        angle2 = round(euler2[2], 4)
        angle3 = round(euler3[2], 4)
        angle4 = round(euler4[2], 4)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        v_state2 = []
        v_state2[:] = self.velodyne_data2[:]
        v_state3 = []
        v_state3[:] = self.velodyne_data3[:]
        v_state4 = []
        v_state4[:] = self.velodyne_data4[:]
        laser_state = [v_state]
        laser_state2 = [v_state2]
        laser_state3 = [v_state3]
        laser_state4 = [v_state4]
        distance = np.linalg.norm(
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )
        distance2 = np.linalg.norm(
            [odom_x2 - self.goal_x2, odom_y2 - self.goal_y2]
        )
        distance3 = np.linalg.norm(
            [odom_x3 - self.goal_x3, odom_y3 - self.goal_y3]
        )
        distance4 = np.linalg.norm(
            [odom_x4 - self.goal_x4, odom_y4 - self.goal_y4]
        )
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
        skew_x2 = self.goal_x2 - odom_x2
        skew_y2 = self.goal_y2 - odom_y2
        skew_x3 = self.goal_x3 - odom_x3
        skew_y3 = self.goal_y3 - odom_y3
        skew_x4 = self.goal_x4 - odom_x4
        skew_y4 = self.goal_y4 - odom_y4
        dot = skew_x * 1 + skew_y * 0
        dot2 = skew_x2 * 1 + skew_y2 * 0
        dot3 = skew_x3 * 1 + skew_y3 * 0
        dot4 = skew_x4 * 1 + skew_y4 * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag12 = math.sqrt(math.pow(skew_x2, 2) + math.pow(skew_y2, 2))
        mag22 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag13 = math.sqrt(math.pow(skew_x3, 2) + math.pow(skew_y3, 2))
        mag23 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        mag14 = math.sqrt(math.pow(skew_x4, 2) + math.pow(skew_y4, 2))
        mag24 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        beta2 = math.acos(dot2 / (mag12 * mag22))
        beta3 = math.acos(dot3 / (mag13 * mag23))
        beta4 = math.acos(dot4 / (mag14 * mag24))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        if skew_y2 < 0:
            if skew_x2 < 0:
                beta2 = -beta2
            else:
                beta2 = 0 - beta2
        if skew_y3 < 0:
            if skew_x3 < 0:
                beta3 = -beta3
            else:
                beta3 = 0 - beta3
        if skew_y4 < 0:
            if skew_x4 < 0:
                beta4 = -beta4
            else:
                beta4 = 0 - beta4
        theta = beta - angle
        theta2 = beta2 - angle2
        theta3 = beta3 - angle3
        theta4 = beta4 - angle4
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        if theta2 > np.pi:
            theta2 = np.pi - theta2
            theta2 = -np.pi - theta2
        if theta2 < -np.pi:
            theta2 = -np.pi - theta2
            theta2 = np.pi - theta2
        if theta3 > np.pi:
            theta3 = np.pi - theta3
            theta3 = -np.pi - theta3
        if theta3 < -np.pi:
            theta3 = -np.pi - theta3
            theta3 = np.pi - theta3
        if theta > np.pi:
            theta4 = np.pi - theta4
            theta4 = -np.pi - theta4
        if theta4 < -np.pi:
            theta4 = -np.pi - theta4
            theta4 = np.pi - theta4
        if distance < GOAL_REACHED:
            done = True
            print('R1 REACHED!')
        if distance2 < GOAL_REACHED:
            done2 = True
            print('R2 REACHED!')
        if distance3 < GOAL_REACHED:
            done3 = True
            print('R3 REACHED!')
        if distance4 < GOAL_REACHED:
            print('R4 REACHED!')
        robot_state = [distance, theta, 0.0, 0.0]
        robot_state2 = [distance2, theta2, 0.0, 0.0]
        robot_state3 = [distance3, theta3, 0.0, 0.0]
        robot_state4 = [distance4, theta4, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        state2 = np.append(laser_state2, robot_state2)
        state3 = np.append(laser_state3, robot_state3)
        state4 = np.append(laser_state4, robot_state4)
        return state, state2, state3, state4, done, done2, done3, done4
    
    def distances(self, n_robots):
        if n_robots == 2:
            x1 = self.last_odom.pose.pose.position.x
            y1 = self.last_odom.pose.pose.position.y
            x2 = self.last_odom2.pose.pose.position.x
            y2 = self.last_odom2.pose.pose.position.y
            d12 = np.linalg.norm(
                    [x1 - x2, y1 - y2]
                )
            return d12, x1, y1, x2, y2
        if n_robots == 3:
            x1 = self.last_odom.pose.pose.position.x
            y1 = self.last_odom.pose.pose.position.y
            x2 = self.last_odom2.pose.pose.position.x
            y2 = self.last_odom2.pose.pose.position.y
            x3 = self.last_odom3.pose.pose.position.x
            y3 = self.last_odom3.pose.pose.position.y
            d12 = np.linalg.norm(
                    [x1 - x2, y1 - y2]
                )
            d13 = np.linalg.norm(
                    [x1 - x3, y1 - y3]
                )
            d23 = np.linalg.norm(
                    [x3 - x2, y3 - y2]
                )
            return d12, d13, d23, x1, y1, x2, y2, x3, y3
        if n_robots == 4:
            x1 = self.last_odom.pose.pose.position.x
            y1 = self.last_odom.pose.pose.position.y
            x2 = self.last_odom2.pose.pose.position.x
            y2 = self.last_odom2.pose.pose.position.y
            x3 = self.last_odom3.pose.pose.position.x
            y3 = self.last_odom3.pose.pose.position.y
            x4 = self.last_odom4.pose.pose.position.x
            y4 = self.last_odom4.pose.pose.position.y
            d12 = np.linalg.norm(
                    [x1 - x2, y1 - y2]
                )
            d13 = np.linalg.norm(
                    [x1 - x3, y1 - y3]
                )
            d14 = np.linalg.norm(
                    [x1 - x4, y1 - y4]
                )
            d23 = np.linalg.norm(
                    [x3 - x2, y3 - y2]
                )
            d24 = np.linalg.norm(
                    [x4 - x2, y4 - y2]
                )
            d34 = np.linalg.norm(
                    [x3 - x4, y3 - y4]
                )
            return d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4
        
    def publish_markers1(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom1"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher1.publish(markerArray)

    def publish_markers2(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom2"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x2
        marker.pose.position.y = self.goal_y2
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher2.publish(markerArray)

    def publish_markers3(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom3"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x3
        marker.pose.position.y = self.goal_y3
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher3.publish(markerArray)

    def publish_markers4(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom4"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x4
        marker.pose.position.y = self.goal_y4
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher4.publish(markerArray)