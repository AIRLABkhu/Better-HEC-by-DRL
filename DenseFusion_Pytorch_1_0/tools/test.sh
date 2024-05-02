#!/bin/sh
i=1

while [ $i -lt 100 ]

do

        python3 /home/airlab/test_ws/src/Better_HEC_by_DRL/DenseFusion_Pytorch_1_0/tools/launch_linemod_ros_auto_discrete_sac.py --num $i
        
        i=$(($i+1))

done