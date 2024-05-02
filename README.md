# Moving End-Effectors by Deep Reinforcement Learning for Better Hand-Eye Calibration Performance 


## 1. Usage

Our code is tested on Ubuntu 20.04 and ROS Noetic.

### 1.1 Build Instructions

- Conda environment
```
conda env create -f hec.yaml
```

### 1.2 Gazebo simulation

If you want to use kinova gen3_lite manipulator, you should follow [ros_kortex](https://github.com/Kinovarobotics/ros_kortex).

- After Build ros_kortex Clone the gazebo repository and catkin make:
'''
mkdir catkin_ws
cd catkin_ws
git clone https://github.com/Seunghui-Shin/Better_HEC_by_DRL.git
mkdir src
cd src
mv -r Better_HEC_by_DRL/ros_kortex src/
cd ~/catkin_ws && catkin_make
'''

### 1.3 Pose Network and Policy
If you want to use DenseFusion for pose network, yo should follow [DenseFusion](https://github.com/j96w/DenseFusion).

- Clone the repository and catkin build:
  
```
mkdir robot_ws
cd robot_ws
mkdir src
cd src
git clone https://github.com/Seunghui-Shin/Better_HEC_by_DRL.git
cd Better_HEC_by_DRL
rm -rf ros_kortex
cd ../..
catkin build
```

## 2. Running the code

### 2.1 Gazebo simulation
```
roslaunch kortex_gazebo spawn_kortex_robot_realsense.launch arm:=gen3_lite
```

### 2.2 Pose Network
```
conda activate hec
cd robot_ws/src/Better_HEC_by_DRL
./DenseFusion_Pytorch_1_0/tools/train.sh
```
   
### 2.3.1 Training
```
conda activate hec
cd robot_ws/src/Better_HEC_by_DRL/hec
python3 python/train_policies/RL_algo_kinova_discrete_sac.py
```
### 2.3.2 Testing
```
conda activate hec
cd robot_ws/src/Better_HEC_by_DRL/hec
python3 python/test_policies/RL_algo_test_kinova_discrete_sac.py
```

## 3. License

License under the 


## 4. Code reference:

Our code is based on the following repositories:

- [DenseFusion](https://github.com/j96w/DenseFusion)
- [Discrete SAC](https://github.com/BY571/SAC_discrete)
- [Learn-to-Calibrate](https://github.com/ethz-asl/Learn-to-Calibrate/tree/master?tab=readme-ov-file)
