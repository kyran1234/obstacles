import rospy
import gym
from gym.utils import seeding
from openai_ros.gazebo_connection import GazeboConnection
from openai_ros.controllers_connection import ControllersConnection
from nav_msgs.msg import Odometry

rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)

odom = None
while odom is None and not rospy.is_shutdown():
    try:
        odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
        rospy.logdebug("Current"+"/odom READY=>")
        print("ojk")
        print(odom)
    except:
        print("no")
        rospy.logerr("Current" +"odom not ready yet, retrying for getting odom")
