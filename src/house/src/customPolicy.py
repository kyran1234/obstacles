from stable_baselines3.common.policies import *
import stable_baselines3.common.policies as common
# from stable_baselines3.common.tf_layers import conv, linear, conv_to_fc, lstm
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import gymnasium as gym


# def modified_shallow_nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#     :param scaled_images: (TensorFlow Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (TensorFlow Tensor) The CNN output layer
#     """
#     activ = tf.nn.relu
#     layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = conv_to_fc(layer_2)
#     return activ(linear(layer_2, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
# 示例：重写浅层CNN（替换 modified_shallow_nature_cnn）
class ModifiedShallowNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 假设输入是图像（例如 (通道数, 高度, 宽度)）
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=2, stride=1),  # 对应原代码的conv层
            nn.ReLU(),  # 对应原代码的tf.nn.relu
            nn.Conv2d(8, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),  # 对应原代码的conv_to_fc（展平为全连接层输入）
        )
        # 计算展平后的维度（需要根据输入图像大小调整，这里假设示例值）
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),  # 对应原代码的linear层
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(observations))


# def modified_deep_nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#     :param scaled_images: (TensorFlow Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (TensorFlow Tensor) The CNN output layer
#     """
#     activ = tf.nn.relu
#     layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=6, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = conv_to_fc(layer_2)
#     layer_3 = activ(linear(layer_2, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
#     return activ(linear(layer_3, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
class ModifiedDeepNatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=6, stride=1),  # 对应原c1层（filter_size=6）
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),  # 对应原c2层（filter_size=3）
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        # 对应原fc1和fc2层
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(observations))


# def tiny_filter_deep_nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#     :param scaled_images: (TensorFlow Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (TensorFlow Tensor) The CNN output layer
#     """
#     activ = tf.nn.relu
#     layer_1 = activ(conv(scaled_images, 'c1', n_filters=6, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = activ(conv(layer_1, 'c2', n_filters=8, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_3 = activ(conv(layer_2, 'c3', n_filters=10, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_4 = activ(conv(layer_3, 'c4', n_filters=12, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_5 = activ(conv(layer_4, 'c5', n_filters=14, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_5 = conv_to_fc(layer_5)
#     layer_6 = activ(linear(layer_5, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
#     layer_7 = activ(linear(layer_6, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
#     return activ(linear(layer_7, 'fc3', n_hidden=128, init_scale=np.sqrt(2)))

class TinyFilterDeepCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 假设输入观测是图像（调整通道数和尺寸以匹配你的实际观测）
        n_input_channels = observation_space.shape[0]  # 根据你的观测空间修改
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(6, 8, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(10, 12, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(12, 14, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),  # 展平特征图
        )
        # 计算CNN输出的维度（根据输入尺寸自动计算）
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(),
                                    nn.Linear(128, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# class CustomShallowCNNPolicy(common.ActorCriticPolicy):
#     
#     def __init__(self, *args, **kwargs):
#         super(CustomShallowCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_shallow_nature_cnn, feature_extraction="cnn")
# 
# 
# class CustomDeepCNNPolicy(common.ActorCriticPolicy):
#     
#     def __init__(self, *args, **kwargs):
#         super(CustomDeepCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_deep_nature_cnn, feature_extraction="cnn")
# 
# class CustomTinyDeepCNNPolicy(common.ActorCriticPolicy):
#     
#     def __init__(self, *args, **kwargs):
#         super(CustomTinyDeepCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=tiny_filter_deep_nature_cnn, feature_extraction="cnn")
# 修复所有自定义策略类，移除feature_extraction参数

class CustomShallowCNNPolicy(common.ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomShallowCNNPolicy, self).__init__(
            *args, **kwargs,
            cnn_extractor=modified_shallow_nature_cnn  # 移除feature_extraction参数
        )

class CustomDeepCNNPolicy(common.ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDeepCNNPolicy, self).__init__(
            *args, **kwargs,
            cnn_extractor=modified_deep_nature_cnn  # 移除feature_extraction参数
        )

class CustomTinyDeepCNNPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            features_extractor_class=TinyFilterDeepCNN,  # 使用自定义的PyTorch特征提取器
            features_extractor_kwargs=dict(features_dim=128),  # 特征维度
            net_arch=dict(pi=[], vf=[]),  # 可以根据需要添加后续全连接层
        )

# 修复CustomMLPPolicy的语法错误
class CustomMLPPolicy(common.ActorCriticPolicy):  # 这里之前是common.，缺少类名
    def __init__(self, *args, **kwargs):
        super(CustomMLPPolicy, self).__init__(
            *args, **kwargs,
            net_arch=[dict(pi=[2880, 1440, 720, 360],
                           vf=[2880, 1440, 720, 360])]  # 移除feature_extraction参数
        )
# class CustomMLPPolicy(common.ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomMLPPolicy, self).__init__(*args, **kwargs,
#                                            net_arch=[dict(pi=[2880, 1440, 720, 360],
#                                                           vf=[2880, 1440, 720, 360])],
#                                            feature_extraction="mlp")