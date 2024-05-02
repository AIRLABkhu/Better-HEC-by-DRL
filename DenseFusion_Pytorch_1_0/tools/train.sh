#!/bin/sh
i=1

while [ $i -lt 1000000000000000 ]

do

        python3 /home/airlab/test_ws/src/Better-HEC-by-DRL/DenseFusion_Pytorch_1_0/tools/launch_linemod_ros_auto_discrete_sac.py --num $i

        i=$(($i+1))

done