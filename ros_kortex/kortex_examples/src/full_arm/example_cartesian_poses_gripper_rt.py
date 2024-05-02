#!/usr/bin/env python3
###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import rospy
import time
from kortex_driver.srv import *
from kortex_driver.msg import *
import tf
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import random

j,k=0,0
lst = []
class ExampleCartesianActionsWithNotifications:
    def __init__(self):
        try:
            rospy.init_node('example_cartesian_poses_with_notifications_python')

            self.HOME_ACTION_IDENTIFIER = 2

            self.action_topic_sub = None
            self.all_notifs_succeeded = True

            self.all_notifs_succeeded = True

            # Get node params
            self.robot_name = rospy.get_param('~robot_name', "my_gen3_lite")

            rospy.loginfo("Using robot_name " + self.robot_name)

            # Init the action topic subscriber
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)
            self.last_action_notif_type = None

            # Init the services
            clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
            rospy.wait_for_service(clear_faults_full_name)
            self.clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

            set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
            rospy.wait_for_service(set_cartesian_reference_frame_full_name)
            self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name, SetCartesianReferenceFrame)

            activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
            rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
            self.activate_publishing_of_action_notification = rospy.ServiceProxy(activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)
            self.listener = tf.TransformListener()
            self.listener.waitForTransform("/camera_depth_optical_frame", "/end_effector_link", rospy.Time(0), rospy.Duration(3.0))
        
        except:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False
            else:
                time.sleep(0.01)

    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            rospy.sleep(2.5)
            return True

    def example_home_the_robot(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()

    def example_set_cartesian_reference_frame(self):
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame()
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            return True

        # Wait a bit
        rospy.sleep(0.25)

    def example_subscribe_to_a_robot_notification(self):
        # Activate the publishing of the ActionNotification
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
            return False
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.sleep(1.0)

        return True
    
    def get_quaternion_rotation_matrix(self, Q_init, switch_w=True):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                This rotation matrix converts a point in the local reference
                frame to a point in the global reference frame.
        """
        # Extract the values from Q

        if switch_w:
            Q = np.insert(Q_init[:3], 0, Q_init[-1])  # put w to first place
        else:
            Q = Q_init  # w already at the first place

        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

        return rot_matrix
    
    def color_img_callback(self,color):
        global j
        image = CvBridge().imgmsg_to_cv2(color, "bgr8")
        (trans,rot) = self.listener.lookupTransform('/camera_depth_optical_frame', '/end_effector_link', rospy.Time(0))
        tvec = np.array([trans[0],trans[1],trans[2]])
        r = np.array([rot[0],rot[1],rot[2],rot[3]])
        R = self.get_quaternion_rotation_matrix(r, switch_w=True)
        cam_to_ee = np.column_stack((R,tvec))
        cam_to_ee= np.row_stack((cam_to_ee, np.array([0.0, 0.0, 0.0, 1.0])))
        print(cam_to_ee)
        lst.append(cam_to_ee)
        cv2.imwrite("/home/airlab/projects/handeye/[8_14]sim_auto_rt/rgb/rgb_{0}.png".format(j),image)
        print("capture rgb!!!")
        j += 1

    def depth_img_callback(self,depth):
        global k
        image = CvBridge().imgmsg_to_cv2(depth, "32SC1")
        image = image.astype(np.uint16)
        cv2.imwrite("/home/airlab/projects/handeye/[8_14]sim_auto_rt/depth/depth_{0}.png".format(k),image)
        print("capture depth!!!")
        k += 1

    def main(self):
        # For testing purposes
        success = self.is_init_success
        try:
            rospy.delete_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python")
        except:
            pass

        if success:
            start = time.time()
            #*******************************************************************************
            # Make sure to clear the robot's faults else it won't move if it's already in fault
            success &= self.example_clear_faults()
            #*******************************************************************************
            
            #*******************************************************************************
            # Start the example from the Home position
            success &= self.example_home_the_robot()
            #*******************************************************************************

            #*******************************************************************************
            # Set the reference frame to "Mixed"
            success &= self.example_set_cartesian_reference_frame()

            #*******************************************************************************
            # Subscribe to ActionNotification's from the robot to know when a cartesian pose is finished
            success &= self.example_subscribe_to_a_robot_notification()

            #*******************************************************************************

            # Prepare and send pose 1
            my_cartesian_speed = CartesianSpeed()
            my_cartesian_speed.translation = 1000 # m/s
            my_cartesian_speed.orientation = 360  # deg/s

            my_constrained_pose = ConstrainedPose()
            my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

            
            for i in range(1,100):
                my_constrained_pose.target_pose.x = 0.3 #random.uniform(0.2,0.4)
                my_constrained_pose.target_pose.y = 0 #random.uniform(-0.2,0.2)
                my_constrained_pose.target_pose.z = 0.2 #random.uniform(0.1,0.3)
                my_constrained_pose.target_pose.theta_x = 0 #random.randint(-45,45)
                my_constrained_pose.target_pose.theta_y = 180 #random.randint(140,220)
                my_constrained_pose.target_pose.theta_z = 0 #random.randint(-45,45)

                req = ExecuteActionRequest()
                req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
                req.input.name = "pose0"
                req.input.handle.action_type = ActionType.REACH_POSE
                req.input.handle.identifier = 1000

                rospy.loginfo("Sending pose 0...")
                self.last_action_notif_type = None

                try:
                    self.execute_action(req)
                except rospy.ServiceException:
                    rospy.logerr("Failed to send pose 0")
                    success = False
                else:
                    rospy.loginfo("Waiting for pose 0 to finish...")

                self.wait_for_action_end_or_abort()

            success &= self.all_notifs_succeeded
            success &= self.all_notifs_succeeded

            end = time.time()
            print(f"{end - start:.5f} sec")
        # For testing purposes
        rospy.set_param("/kortex_examples_test_results/cartesian_poses_with_notifications_python", success)
        
        print(len(lst))

        if not success:
            rospy.logerr("The example encountered an error.")

if __name__ == "__main__":

    ex = ExampleCartesianActionsWithNotifications()
    ex.main()

