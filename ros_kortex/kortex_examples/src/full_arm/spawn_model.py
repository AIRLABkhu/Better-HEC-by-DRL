#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from gazebo_msgs.srv import DeleteModel
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Pose, Quaternion
import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to Quaternion
    """
    quaternion = Quaternion()
    quaternion.x = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    quaternion.y = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    quaternion.z = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    quaternion.w = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return quaternion


def spawn_model():
    rospy.init_node('spawn_model_node')

    # 모델을 스폰하기 전에 Gazebo 서비스 클라이언트를 만듭니다.
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

    # robot_description 매개변수 가져오기
    robot_description = rospy.get_param('robot_description')

    # 모델 초기 위치 및 자세 설정
    model_pose = Pose()
    model_pose.position.x = 0.3
    model_pose.position.y = -1.0
    model_pose.position.z = 0.2  # 원하는 고도로 조정

    roll = 0.0  # 원하는 롤 각도 (radians)로 조정
    pitch = 0.0  # 원하는 피치 각도 (radians)로 조정
    yaw = 1.5707  # 원하는 요우 각도 (radians)로 조정
    quaternion = euler_to_quaternion(roll, pitch, yaw)
    model_pose.orientation = quaternion

    # 스폰할 모델의 정보 설정
    model_name = "realsense"  # 모델의 이름으로 바꿔주세요
    model_request = SpawnModelRequest()
    model_request.model_name = model_name
    model_request.model_xml = robot_description  # robot_description 매개변수를 사용
    model_request.initial_pose = model_pose

    try:
        # 모델 스폰 서비스 호출
        response = spawn_model_client(model_request)
        rospy.loginfo("Model %s spawned successfully!", model_name)
    except rospy.ServiceException as e:
        rospy.logerr("Model spawn failed: %s", e)


def delete_model():

    # 모델 이름 설정 (삭제할 모델의 이름으로 변경)
    model_name = "realsense"

    try:
        # Gazebo의 DeleteModel 서비스 클라이언트 생성
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        # 모델 삭제 서비스 호출
        delete_model(model_name)
        rospy.loginfo("Model %s deleted successfully!", model_name)
    except rospy.ServiceException as e:
        rospy.logerr("Model deletion failed: %s", e)

if __name__ == '__main__':
    delete_model()
    spawn_model()