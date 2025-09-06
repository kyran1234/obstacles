#!/usr/bin/env python3
import rospy
import random
from geometry_msgs.msg import WrenchStamped  # 用于发布力/力矩
from gazebo_msgs.msg import ContactsState

class CatForceController:
    def __init__(self):
        rospy.init_node('cat_controller', anonymous=True)
        
        # 1. 发布力和力矩的话题（与world中插件配置一致）
        self.force_pub = rospy.Publisher('/cat/apply_force', WrenchStamped, queue_size=10)
        self.torque_pub = rospy.Publisher('/cat/apply_torque', WrenchStamped, queue_size=10)
        
        # 2. 订阅碰撞话题
        self.collision_sub = rospy.Subscriber('/cat/collision', ContactsState, self.collision_cb)
        
        # 控制参数
        self.rate = rospy.Rate(10)  # 10Hz控制频率
        self.force_range = [0.5, 1.5]  # 随机线性力范围（N），适配1kg质量
        self.torque_range = [-0.8, 0.8]  # 随机力矩范围（N·m），控制转向
        self.collision_flag = False  # 碰撞标志
        self.collision_recovery_time = 1.0  # 碰撞后恢复时间（秒）
        self.last_collision_time = 0  # 上次碰撞时间

        rospy.loginfo("猫力控控制器启动，开始随机运动...")

    def collision_cb(self, data):
        """碰撞回调：检测到碰撞时设置标志"""
        if data.states and "ground_plane" not in data.states[0].collision1_name:
            self.collision_flag = True
            self.last_collision_time = rospy.get_time()
            rospy.loginfo("检测到碰撞，开始避障...")

    def generate_random_force(self):
        """生成随机线性力（沿x轴，驱动前进）"""
        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        # 碰撞时反向施力（后退），否则随机前进力
        if self.collision_flag:
            wrench.wrench.force.x = -random.uniform(*self.force_range) * 1.2  # 反向力更大
        else:
            wrench.wrench.force.x = random.uniform(*self.force_range)
        # y/z轴力为0，避免上下/左右偏移
        wrench.wrench.force.y = 0.0
        wrench.wrench.force.z = 0.0
        return wrench

    def generate_random_torque(self):
        """生成随机力矩（沿z轴，控制转向）"""
        wrench = WrenchStamped()
        wrench.header.stamp = rospy.Time.now()
        # 碰撞时随机转向，否则定期改变方向
        if self.collision_flag:
            wrench.wrench.torque.z = random.uniform(*self.torque_range) * 1.5  # 碰撞后转向幅度更大
        else:
            # 每3秒随机改变一次转向
            if rospy.get_time() - self.last_collision_time > 3.0:
                wrench.wrench.torque.z = random.uniform(*self.torque_range)
            else:
                wrench.wrench.torque.z = 0.0  # 未碰撞时保持直线
        return wrench

    def run(self):
        """主循环：发布力/力矩指令"""
        while not rospy.is_shutdown():
            # 检查碰撞恢复时间：超过1秒后取消碰撞标志
            if rospy.get_time() - self.last_collision_time > self.collision_recovery_time:
                self.collision_flag = False

            # 生成并发布力和力矩
            force_msg = self.generate_random_force()
            torque_msg = self.generate_random_torque()
            self.force_pub.publish(force_msg)
            self.torque_pub.publish(torque_msg)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = CatForceController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("控制器中断")
    except Exception as e:
        rospy.logerr(f"控制器错误: {str(e)}")