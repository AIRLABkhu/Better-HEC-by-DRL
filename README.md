# Moving End-Effectors by Deep Reinforcement Learning for Better Hand-Eye Calibration Performance 


## 1. Usage

Our code is tested on Ubuntu 20.04 and ROS Noetic.

1.1 Build Instructions

- Conda environment
```
conda env create -f hec.yaml
```

- Gazebo simulation
if you want to use kinova gen3_lite manipulator, you should follow [ros_kortex](https://github.com/Kinovarobotics/ros_kortex)

- After Build ros_kortex Clone the gazebo repository and catkin make:
'''
mkdir catkin_ws
cd catkin_ws
mkdir src
cd src
git clone 
cd ~/catkin_ws && catkin_make
'''

- Clone the repository and catkin build:
```
mkdir robot_ws
cd robot_ws
mkdir src
cd src
git clone https://github.com/Seunghui-Shin/Better-HEC-by-DRL.git
cd ../..
catkin build
```
## 2. Running the code
   
### 2.1 Training
```
cd robot_ws/src/hec
python3 python/train_policies/RL_algo_kinova_discrete_sac.py
```
### 2.2 Testing
```
cd robot_ws/src/hec
python3 python/test_policies/RL_algo_test_kinova_discrete_sac.py
```

## 3. Code reference:

Our code is based on the following repositories:

- [Discrete SAC](https://github.com/BY571/SAC_discrete)
- [Learn-to-Calibrate](https://github.com/ethz-asl/Learn-to-Calibrate/tree/master?tab=readme-ov-file)
