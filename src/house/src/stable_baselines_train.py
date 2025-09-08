#!/usr/bin/env python

import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
ros_path = '/opt/ros/neotic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
sys.path.append('/opt/ros/neotic/lib/python3/dist-packages')
from openai_ros.task_envs.turtlebot2.turtlebot2_maze import TurtleBot2MazeEnv
import rospy
import os
from customPolicy import *
# Custom MLP policy of three layers of size 128 each
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[144, 144, 144],
                                                          vf=[144, 144, 144])],
                                           feature_extraction="mlp")

if __name__ == '__main__':
	world_file = sys.argv[1]
	number_of_robots = sys.argv[2]
	rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)
	env_temp = TurtleBot2MazeEnv
	env = DummyVecEnv([lambda k=k:env_temp(world_file, k) for k in range(int(number_of_robots))])
	model = PPO(CustomTinyDeepCNNPolicy, env, n_steps=900, ent_coef=0.01, learning_rate=0.0001, batch_size=180, n_epochs=10, tensorboard_log="../PPO2_turtlebot_tensorboard/", verbose=1)
	model.learn(total_timesteps=1200000)
	model.save("ppo2_turtlebot")
																																																																																																																																																											