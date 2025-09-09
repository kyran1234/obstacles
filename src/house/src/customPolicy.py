from stable_baselines3.common.policies import *
import stable_baselines3.common.policies as common
# from stable_baselines3.common.tf_layers import conv, linear, conv_to_fc, lstm
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# class TinyFilterDeepCNN(BaseFeaturesExtractor):
#     """
#     严格匹配原TensorFlow版本的PyTorch特征提取器
#     保持卷积核大小、通道数、全连接层维度与原代码一致
#     """
#     def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
#         super().__init__(observation_space, features_dim)
# 
#         # 输入图像通道数（原代码中为scaled_images的通道数）
#         n_input_channels = observation_space.shape[2] # 4
#         # 卷积层序列（严格对应原TensorFlow的5层卷积）
#         self.cnn = nn.Sequential(
#             # layer_1: c1卷积层 (6个2x2卷积核)
#             nn.Conv2d(
#                 in_channels=n_input_channels,
#                 out_channels=6,
#                 kernel_size=2,
#                 stride=1,
#                 padding=0  # 原TensorFlow默认无padding
#             ),
#             nn.ReLU(),  # 对应tf.nn.relu
#             
#             # layer_2: c2卷积层 (8个2x2卷积核)
#             nn.Conv2d(
#                 in_channels=6,
#                 out_channels=8,
#                 kernel_size=2,
#                 stride=1,
#                 padding=0
#             ),
#             nn.ReLU(),
#             
#             # layer_3: c3卷积层 (10个2x2卷积核)
#             nn.Conv2d(
#                 in_channels=8,
#                 out_channels=10,
#                 kernel_size=2,
#                 stride=1,
#                 padding=0
#             ),
#             nn.ReLU(),
#             
#             # layer_4: c4卷积层 (12个3x3卷积核)
#             nn.Conv2d(
#                 in_channels=10,
#                 out_channels=12,
#                 kernel_size=3,
#                 stride=1,
#                 padding=0
#             ),
#             nn.ReLU(),
#             
#             # layer_5: c5卷积层 (14个3x3卷积核)
#             nn.Conv2d(
#                 in_channels=12,
#                 out_channels=14,
#                 kernel_size=3,
#                 stride=1,
#                 padding=0
#             ),
#             nn.ReLU(),
#             
#             # 对应原代码的conv_to_fc（展平特征图）
#             nn.Flatten()
#         )
#         
#         # 计算卷积层输出的扁平化维度（确保与原代码展平后尺寸一致）
#         with torch.no_grad():
#             # 获取观测空间样本并添加批次维度
#             sample = observation_space.sample()[None, ...]  # 形状: (1, C, H, W)
#             # 计算卷积层输出尺寸
#             cnn_output = self.cnn(torch.as_tensor(sample, dtype=torch.float32))
#             n_flatten = cnn_output.shape[1]  # 扁平化后的维度
#         
#         # 全连接层序列（对应原TensorFlow的fc1、fc2、fc3）
#         self.fc = nn.Sequential(
#             # layer_6: fc1 (256隐藏单元)
#             nn.Linear(in_features=n_flatten, out_features=256),
#             nn.ReLU(),
#             
#             # layer_7: fc2 (128隐藏单元)
#             nn.Linear(in_features=256, out_features=128),
#             nn.ReLU(),
#             
#             # 输出层: fc3 (128隐藏单元，与原代码保持一致)
#             nn.Linear(in_features=128, out_features=features_dim),
#             nn.ReLU()
#         )
#         
#         # 初始化权重（匹配原代码的init_scale=np.sqrt(2)）
#         self._initialize_weights()
# 
#     def _initialize_weights(self):
#         """初始化权重，匹配原TensorFlow的init_scale=np.sqrt(2)"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 # 原代码使用init_scale=np.sqrt(2)，对应PyTorch的Xavier初始化变种
#                 nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
# 
#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         """前向传播，返回提取的特征"""
#         return self.fc(self.cnn(observations))
class TinyFilterDeepCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # 保存输入尺寸
        self.input_height = observation_space.shape[1]
        self.input_width = observation_space.shape[2]
        
        # 计算卷积层输出尺寸
        def conv2d_size_out(size, kernel_size=1, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        # 计算经过所有卷积层后的尺寸
        conv_c = observation_space.shape[0]
        conv_w = self.input_width
        conv_h = self.input_height
        
        # 卷积层定义
        self.conv1 = nn.Conv2d(self.input_width, 6, kernel_size=2, stride=1)
        conv_c = conv2d_size_out(conv_c, 2)
        conv_h = conv2d_size_out(conv_h, 2)
        
        self.conv2 = nn.Conv2d(6, 8, kernel_size=2, stride=1)
        conv_c = conv2d_size_out(conv_c, 2)
        conv_h = conv2d_size_out(conv_h, 2)
        
        self.conv3 = nn.Conv2d(8, 10, kernel_size=2, stride=1)
        conv_c = conv2d_size_out(conv_c, 2)
        conv_h = conv2d_size_out(conv_h, 2)
        
        self.conv4 = nn.Conv2d(10, 12, kernel_size=3, stride=1)
        conv_c = conv2d_size_out(conv_c, 3)
        conv_h = conv2d_size_out(conv_h, 3)
        
        self.conv5 = nn.Conv2d(12, 14, kernel_size=3, stride=1)
        conv_c = conv2d_size_out(conv_c, 3)
        conv_h = conv2d_size_out(conv_h, 3)
        
        # 计算全连接层输入尺寸
        self.flat_size = 14 * conv_c * conv_h
        
        # 全连接层定义
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, features_dim)

        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # 使用与原始代码相同的初始化方式
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 处理输入维度 (batch_size, height, width) -> (batch_size, 1, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # 添加通道维度
        
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

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