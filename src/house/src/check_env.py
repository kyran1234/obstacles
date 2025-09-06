#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/neotic/lib/python3/dist-packages')
import gym
from stable_baselines3.common.env_checker import check_env
sys.path.append('/opt/ros/neotic/lib/python3/dist-packages')
from openai_ros.task_envs.turtlebot2 import turtlebot2_maze
import rospy



rospy.init_node('turtlebot_gym', anonymous=True, log_level=rospy.WARN)
env = gym.make('TurtleBot2Maze-v0')
check_env(env)

