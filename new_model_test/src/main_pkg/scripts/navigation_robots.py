#!/usr/bin/env python3 

import time
from multi_navigation import GazeboEnv, TD3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from spawning_robots import robots_spawner
from anti_collision import anti_collision

package = "main_pkg"
launch_file = "main_launch.launch"
file_name = "TD3"
expl_noise = 1
robot_dim = 4
env_dim = 20
state_dim = env_dim + robot_dim
action_dim = 2
max_action = 0.5
epsilom = 0.9
n_robots = int(input("Give the number of robots you'd like to spawn[1-4]:"))
spawn = robots_spawner(n_robots)
spawn.spawning()
time.sleep(10)
env = GazeboEnv(n_robots, env_dim)
print("the environment is set!")
time.sleep(10)
network = TD3(state_dim, action_dim, max_action)
network.load("/home/wsl-ros/new_model_test/src/main_pkg/scripts/pytorch_models", file_name)
if n_robots == 1: 
    rospy.wait_for_message("/goal", PoseStamped)
    print("A new goal is received!")
    state = env.state()
    done = False
    mission_finished = False
    while not mission_finished:
        while not done:
            action = network.get_action(state)
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                -max_action, max_action
            )
            a_in = [(action[0]+1)/2, 2*action[1]]
            next_state, done = env.step(a_in)
            state = next_state
        continue_mission = int(input("If you want to asign another goal, type 1: "))
        if continue_mission == 1:
            rospy.wait_for_message("/goal", PoseStamped)
            print("A new goal is received!")
            state = env.state()
            done = False
        else: mission_finished = True
if n_robots == 2:
    rospy.wait_for_message("/goal", PoseStamped)
    rospy.wait_for_message("/goal2", PoseStamped)
    print("New goals are received!")
    state, state2 = env.state2()
    done = False
    done2 = False
    mission_finished = False
    while not mission_finished:
        while not done or not done2:
            d12, x1, y1, x2, y2 = env.distances(n_robots)
            real_g1x = env.goal_x
            real_g1y = env.goal_y
            if d12 >= epsilom:
                env.goal_x = real_g1x
                env.goal_y = real_g1y
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                next_state, next_state2, done, done2 = env.step2(a_in, a_in2)
                state = next_state
                state2 = next_state2
                d12, x1, y1, x2, y2 = env.distances(n_robots)
            else:
                warning = anti_collision(x1, y1, x2, y2)
                g1x, g1y = warning.avoiding_collisions()
                env.goal_x = g1x
                env.goal_y = g1y
                env.goal_x2 = x2
                env.goal_y2 = y2
                action = network.get_action(state)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [0.0, 0.0]
                print("COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, done, done2 = env.step2(a_in, a_in2)
                state1 = next_state
                done1 = False
                done2 = False
                state2 = next_state2
                d12, x1, y1, x2, y2 = env.distances(n_robots)
        continue_mission = int(input("If you want to asign other goals, type 1: "))
        if continue_mission == 1:
            rospy.wait_for_message("/goal", PoseStamped)
            rospy.wait_for_message("/goal2", PoseStamped)
            print("New goals are received!")
            state, state2 = env.state2()
            done = False
            done2 = False
        else: mission_finished = True
if n_robots == 3:
    rospy.wait_for_message("/goal", PoseStamped)
    rospy.wait_for_message("/goal2", PoseStamped)
    rospy.wait_for_message("/goal3", PoseStamped)
    print("New goals are received!")
    state, state2, state3 = env.state3()
    done = False
    done2 = False
    done3 = False
    mission_finished = False
    while not mission_finished:
        while not done or not done2 or not done3:
            d12, d13, d23, x1, y1, x2, y2, x3, y3 = env.distances(n_robots)
            real_g1x = env.goal_x
            real_g1y = env.goal_y
            real_g2x = env.goal_x2
            real_g2y = env.goal_y2
            real_g3x = env.goal_x3
            real_g3y = env.goal_y3 
            if d12 < epsilom:
                warning = anti_collision(x1, y1, x2, y2)
                g1x, g1y = warning.avoiding_collisions()
                env.goal_x = g1x
                env.goal_y = g1y
                env.goal_x2 = x2
                env.goal_y2 = y2
                action = network.get_action(state)
                action3 = network.get_action(state3)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [0.0, 0.0]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                print("R1-R2 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, done, done2, done3 = env.step3(a_in, a_in2, a_in3)                
                state1 = next_state
                state3 = next_state3
                done1 = False
                done2 = False
                done3 = False
                state2 = next_state2
                d12, d13, d23, x1, y1, x2, y2, x3, y3 = env.distances(n_robots)
            elif d13 < epsilom:
                warning = anti_collision(x1, y1, x3, y3)
                g1x, g1y = warning.avoiding_collisions()
                env.goal_x = g1x
                env.goal_y = g1y
                env.goal_x3 = x3
                env.goal_y3 = y3
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in3 = [0.0, 0.0]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                print("R1-R3 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, done, done2, done3 = env.step3(a_in, a_in2, a_in3)
                state1 = next_state
                state2 = next_state2
                done1 = False
                done2 = False
                done3 = False
                state3 = next_state3
                d12, d13, d23, x1, y1, x2, y2, x3, y3 = env.distances(n_robots)
            elif d23 < epsilom:
                warning = anti_collision(x2, y2, x3, y3)
                g2x, g2y = warning.avoiding_collisions()
                env.goal_x2 = g2x
                env.goal_y2 = g2y
                env.goal_x3 = x3
                env.goal_y3 = y3
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in3 = [0.0, 0.0]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                print("R2-R3 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, done, done2, done3 = env.step3(a_in, a_in2, a_in3)
                state1 = next_state
                state2 = next_state2
                done1 = False
                done2 = False
                done3 = False
                state3 = next_state3
                d12, d13, d23, x1, y1, x2, y2, x3, y3 = env.distances(n_robots)
            else:
                env.goal_x = real_g1x
                env.goal_y = real_g1y
                env.goal_x2 = real_g2x
                env.goal_y2 = real_g2y
                env.goal_x3 = real_g3x
                env.goal_y3 = real_g3y
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action3 = network.get_action(state3)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                next_state, next_state2, next_state3, done, done2, done3 = env.step3(a_in, a_in2, a_in3)
                state = next_state
                state2 = next_state2
                state3 = next_state3
                d12, d13, d23, x1, y1, x2, y2, x3, y3 = env.distances(n_robots)
        continue_mission = int(input("If you want to asign other goals, type 1: "))
        if continue_mission == 1:
            rospy.wait_for_message("/goal", PoseStamped)
            rospy.wait_for_message("/goal2", PoseStamped)
            rospy.wait_for_message("/goal3", PoseStamped)
            print("New goals are received!")
            state, state2, state3 = env.state3()
            done = False
            done2 = False
            done3 = False
        else: mission_finished = True
if n_robots == 4:
    rospy.wait_for_message("/goal", PoseStamped)
    rospy.wait_for_message("/goal2", PoseStamped)
    rospy.wait_for_message("/goal3", PoseStamped)
    rospy.wait_for_message("/goal4", PoseStamped)
    print("New goals are received!")
    state, state2, state3, state4 = env.state4()
    done = False
    done2 = False
    done3 = False
    done4 = False
    mission_finished = False
    while not mission_finished:
        while not done or not done2 or not done3 or not done4:
            d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
            real_g1x = env.goal_x
            real_g1y = env.goal_y
            real_g2x = env.goal_x2
            real_g2y = env.goal_y2
            real_g3x = env.goal_x3
            real_g3y = env.goal_y3
            real_g4x = env.goal_x4
            real_g4y = env.goal_y4
            if d12 < epsilom: 
                warning = anti_collision(x1, y1, x2, y2)
                g1x, g1y = warning.avoiding_collisions()
                env.goal_x = g1x
                env.goal_y =  g1y
                env.goal_x2 = x2
                env.goal_y2 = y2
                action = network.get_action(state)
                action3 = network.get_action(state3)
                action4 = network.get_action(state4)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action4 = (action4 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [0.0, 0.0]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                a_in4 = [(action4[0]+1)/2, 2*action4[1]]
                print("R1-R2 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)                
                state1 = next_state
                state3 = next_state3
                state4 = next_state4
                done1 = False
                done2 = False
                done3 = False
                done4 = False
                state2 = next_state2
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
            elif d13 < epsilom:
                warning = anti_collision(x1, y1, x3, y3)
                g1x, g1y = warning.avoiding_collisions()
                env.goal_x = g1x
                env.goal_y =  g1y
                env.goal_x3 = x3
                env.goal_y3 = y3
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action4 = network.get_action(state4)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action4 = (action4 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [0.0, 0.0]
                a_in4 = [(action4[0]+1)/2, 2*action4[1]]
                print("R1-R3 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)                
                state1 = next_state
                state2 = next_state2
                state4 = next_state4
                done1 = False
                done2 = False
                done3 = False
                done4 = False
                state3 = next_state3
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
            elif d14 < epsilom:
                warning = anti_collision(x1, y1, x4, y4)
                g1x, g1y = warning.avoiding_collisions()
                env.goal_x = g1x
                env.goal_y =  g1y
                env.goal_x4 = x4
                env.goal_y4 = y4
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action3= network.get_action(state3)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                a_in4 = [0.0, 0.0]            
                print("R1-R4 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)                
                state1 = next_state
                state2 = next_state2
                state3 = next_state3
                done1 = False
                done2 = False
                done3 = False
                done4 = False
                state4 = next_state4
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
            elif d23 < epsilom:
                warning = anti_collision(x2, y2, x3, y3)
                g2x, g2y = warning.avoiding_collisions()
                env.goal_x2 = g2x
                env.goal_y2 = g2y
                env.goal_x3 = x3
                env.goal_y3 = y3
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action4 = network.get_action(state4)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action4 = (action4 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [0.0, 0.0]
                a_in4 = [(action4[0]+1)/2, 2*action4[1]]
                print("R1-R3 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)                
                state1 = next_state
                state2 = next_state2
                state4 = next_state4
                done1 = False
                done2 = False
                done3 = False
                done4 = False
                state3 = next_state3
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
            elif d24 < epsilom:
                warning = anti_collision(x2, y2, x4, y4)
                g2x, g2y = warning.avoiding_collisions()
                env.goal_x2 = g2x
                env.goal_y2 = g2y
                env.goal_x4 = x4
                env.goal_y4 = y4
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action3= network.get_action(state3)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                a_in4 = [0.0, 0.0]            
                print("R2-R4 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)                
                state1 = next_state
                state2 = next_state2
                state3 = next_state3
                done1 = False
                done2 = False
                done3 = False
                done4 = False
                state4 = next_state4
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
            elif d34 < epsilom:
                warning = anti_collision(x2, y2, x4, y4)
                g3x, g3y = warning.avoiding_collisions()
                env.goal_x3 = g3x
                env.goal_y3 = g3y
                env.goal_x4 = x4
                env.goal_y4 = y4
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action3= network.get_action(state3)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in1 = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                a_in4 = [0.0, 0.0]            
                print("R3-R4 COLLISION AVOIDANCE ACTIVATED!")
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)                
                state1 = next_state
                state2 = next_state2
                state3 = next_state3
                done1 = False
                done2 = False
                done3 = False
                done4 = False
                state4 = next_state4
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)           
            else:
                env.goal_x = real_g1x
                env.goal_y = real_g1y
                env.goal_x2 = real_g2x
                env.goal_y2 = real_g2y
                env.goal_x3 = real_g3x
                env.goal_y3 = real_g3y
                env.goal_x4 = real_g4x
                env.goal_y4 = real_g4y
                action = network.get_action(state)
                action2 = network.get_action(state2)
                action3 = network.get_action(state3)
                action4 = network.get_action(state4)
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action2 = (action2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action3 = (action3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                action4 = (action4 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                    -max_action, max_action
                )
                a_in = [(action[0]+1)/2, 2*action[1]]
                a_in2 = [(action2[0]+1)/2, 2*action2[1]]
                a_in3 = [(action3[0]+1)/2, 2*action3[1]]
                a_in4 = [(action4[0]+1)/2, 2*action4[1]]
                next_state, next_state2, next_state3, next_state4, done, done2, done3, done4 = env.step4(a_in, a_in2, a_in3, a_in4)
                state = next_state
                state2 = next_state2
                state3 = next_state3
                state4 = next_state4
                d12, d13, d14, d23, d24, d34, x1, y1, x2, y2, x3, y3, x4, y4 = env.distances(n_robots)
        continue_mission = int(input("If you want to asign other goals, type 1: "))
        if continue_mission == 1:
            rospy.wait_for_message("/goal", PoseStamped)
            rospy.wait_for_message("/goal2", PoseStamped)
            rospy.wait_for_message("/goal3", PoseStamped)
            rospy.wait_for_message("/goal4", PoseStamped)
            print("New goals are received!")
            state, state2, state3, state4 = env.state4()
            done = False
            done2 = False
            done3 = False
            done4 = False
        else: mission_finished = True
print("Mission accomplished!")     
rospy.signal_shutdown("We did it!")