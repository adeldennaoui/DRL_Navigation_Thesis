# DRL_Navigation_Thesis
Summary: A repository that keeps track on my thesis work on implementing a deep reinforcement learning framework for the navigation of a MRS without the need of a map.

This work is mainly inspired and based on the repository (reiniscimurs/DRL-robot-navigation): https://github.com/reiniscimurs/DRL-robot-navigation
The repository is divided in 2 worspaces:
1. DRL-robot-navigation (which is cloned from the above repository): the steps to take into consideration to launch the DRL training/testing is mentioned in the worspace's README.txt
2. model_test: this workspace is where I am experimenting, based on the first workspace's framework, the uses of the trained model - especially in the case of a MRS.

My setups: 
ROS Noetic
Ubuntu 20.04 (I am using a VM using 75% of my RAM which is ~12GB and all 4 cores, total storage of VM: 20GB)
the device used for training in my case: CPU intel I7 (IMPORTANT: go to the 'problems I faced' section at the bottom)
Tensorboard 2.11.2 (installation: pip install tensorboard)
Pytorch 1.9.1+cpu (installation:  pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html)

Notes:
- In case of some Gazebo models don't exist in your GAZEBO_MODEL_PATH:
1. run this command in /home/<your_home_dir>: git clone https://github.com/osrf/gazebo_models
2. check in TD3.world (it's in ~/DRL_Navigation_Thesis/DRL-robot-navigation/catkinb_ws/src/multi_robot_scenario/launch), the models that are missing and then copy/paste folders of those models from ~/gazebo_models to ~/DRL-robot-navigation/catkinb_ws/src/multi_robot_scenario/models and ~/model_test/src/main_pkg/models

- Follow these steps for launching the stuff model_test:
1. cd ~/DRL_Navigation_thesis/model_test
2. catkin_make_isolated (the errors you may face are PROBABLY due to missing dependencies, you should know what's missing from the errors' text).
3. setup resources:
4. export ROS_HOSTNAME=localhost
5. export ROS_MASTER_URI=http://localhost:11311
6. export ROS_PORT_SIM=11311
7. export GAZEBO_RESOURCE_PATH=~/DRL_Navigation_thesis/model_test/src/main_pkg/launch
8. source ~/.bashrc
9. source ~/DRL_Navigation_thesis/model_test/devel_isolated/setup.bash
10. cd ~/DRL_Navigation_thesis/model_test/src/main_pkg/scripts
11. python3 <SCRIPT_OF_INTEREST>

- If you pay attention, the original authors are using the Velodyne laser scanners to sense the 20 laser states, whilst I use the normal built-in LiDAR. It is mainly because subscribing to the original LiDAR and taking the information I need is easier for me to code and in theory, the performance should be very similar if not the same. Things work well in the first script.

- I will keep updating the repository.

The scripts that I've written so far:
- navigation.py: When this script is run, the robot will wait for the user to send a geometry_msg/PoseStamped message (topic: /move_base_simple/goal) and the robot will try to reach the global using the trained model by dividing the road into multiple mini-goals (in the script, I divided the road into 3 goals, you can change it). In order to test Aalekh (a colleague)'s work with the model I trained, I asked Aalekh to send me the message through Node-Red. His VM is connected to the cloud and the cloud is connected to my VM using the MQTT protocol) and the test works!
- multirobot_navigation.py: I am still working on it - IT HAS ERRORS. What I want to understand from this script is how would the robots behave when they are given random goals in the space they share and they have to navigate to them using the model I trained. P.S: The range of the LiDAR on the robots cover the whole room they're sharing so keep in mind that both robots are constantly sensing a moving/dynamic obstacle (which is the other robot).

Problems I faced:
- Using a VM and having to train the model using a CPU: It took the original authors of the DRL method hours to reach reach the desired behavior (convergence) while it took me days to start noticing a desirable behavior (the used trained model TD3_actor.pth in ~/DRL_Navigation_thesis/model_test/src/main_pkg/scripts/pytorch_models is hardly ideal,  it does the job but I recommend replacing it with the model you train if your setup is better). I think the bigger blame is for using the CPU instead of the GPU (While CPUs can process many general tasks in a fast, sequential manner, GPUs use parallel computing to break down massively complex problems into multiple smaller simultaneous calculations. This makes them ideal for handling the massively distributed computational processes required for machine learning - our neural network has around ~2,230,000). I recommend installing CUDA drivers on your VM if you have space or straightup use CUDA on your host PC if it already uses Ubuntu.
- Another problem that could accure if you're using a similar setup to mine, you may notice the robot walking in circles. It means that the training is going nowhere. Just retart it and it should work at some point. I heard that changing the default seed in the script could work (I am using seed=0, aka PREDICTABLE random numbers).
