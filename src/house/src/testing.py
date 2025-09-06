import rospy
import gym
from gym.utils import seeding
from openai_ros.gazebo_connection import GazeboConnection
from openai_ros.controllers_connection import ControllersConnection

initial_pose = {}
initial_pose["x_init"] = 0.0
initial_pose["y_init"] = 0.0
initial_pose["z_init"] = 0.0
initial_pose["roll_init"] = 0.0
initial_pose["pitch_init"] = 0.0
initial_pose["yaw_init"] = 0.0
initial_pose["x_rot_init"] = 0.0
initial_pose["y_rot_init"] = 0.0
initial_pose["z_rot_init"] = 0.0
initial_pose["w_rot_init"] = 1.0
rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)
gazebo = GazeboConnection(True, 0, initial_pose = initial_pose, reset_world_or_sim="ROBOT")
controllers_object = ControllersConnection(namespace="", controllers_list={})

odom = None
rospy.logdebug("Waiting for "+"odom state to be READY...")
while odom is None and not rospy.is_shutdown():
    try:
        odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
        rospy.logdebug("Current"+"/odom READY=>")

    except:
        rospy.logerr("Current" +"odom not ready yet, retrying for getting odom")
