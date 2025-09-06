#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

def odom_callback(msg):
    """里程计消息回调函数：打印位置、姿态、速度"""
    # 1. 提取位置信息（x/y坐标，单位：米）
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    # 2. 提取姿态信息（四元数转偏航角）
    orientation_q = msg.pose.pose.orientation
    (roll, pitch, yaw) = euler_from_quaternion([
        orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
    ])
    # 3. 提取速度信息
    linear_vel_x = msg.twist.twist.linear.x
    angular_vel_z = msg.twist.twist.angular.z

    # 每0.5秒打印一次，避免刷屏
    if rospy.Time.now() - odom_callback.last_print_time > rospy.Duration(0.5):
        rospy.loginfo("="*50)
        rospy.loginfo("【里程计信息】")
        rospy.loginfo(f"位置：x={x:.2f} m, y={y:.2f} m")
        rospy.loginfo(f"姿态：偏航角={yaw:.2f} rad（约{yaw*180/3.1416:.1f}°）")
        rospy.loginfo(f"速度：线速度={linear_vel_x:.2f} m/s, 角速度={angular_vel_z:.2f} rad/s")
        odom_callback.last_print_time = rospy.Time.now()

def scan_callback(msg):
    """激光雷达消息回调函数：打印扫描范围、关键距离"""
    # 1. 提取激光雷达基础参数
    angle_min = msg.angle_min
    angle_max = msg.angle_max
    angle_increment = msg.angle_increment
    range_min = msg.range_min
    range_max = msg.range_max
    ranges = msg.ranges

    # 2. 提取关键角度的距离（正前、正左、正右）
    front_idx = len(ranges) // 2
    front_range = ranges[front_idx] if ranges[front_idx] < range_max else float('inf')
    left_idx = len(ranges) // 4
    left_range = ranges[left_idx] if ranges[left_idx] < range_max else float('inf')
    right_idx = 3 * len(ranges) // 4
    right_range = ranges[right_idx] if ranges[right_idx] < range_max else float('inf')

    # 每1秒打印一次，避免刷屏
    if rospy.Time.now() - scan_callback.last_print_time > rospy.Duration(1.0):
        rospy.loginfo("="*50)
        rospy.loginfo("【激光雷达信息】")
        rospy.loginfo(f"扫描范围：{angle_min:.2f} rad ~ {angle_max:.2f} rad（约{angle_min*180/3.1416:.1f}°~{angle_max*180/3.1416:.1f}°）")
        rospy.loginfo(f"角度分辨率：{angle_increment:.4f} rad/步（约{angle_increment*180/3.1416:.2f}°/步）")
        rospy.loginfo(f"探测距离范围：{range_min:.2f} m ~ {range_max:.2f} m")
        rospy.loginfo(f"关键角度距离：正前方={front_range:.2f} m, 正左方={left_range:.2f} m, 正右方={right_range:.2f} m")
        scan_callback.last_print_time = rospy.Time.now()

def main():
    # 1. 初始化ROS节点（必须先执行，启动时间系统）
    rospy.init_node('mybot_sensor_listener', anonymous=True)
    rospy.loginfo("已启动 mybot 传感器监听节点，开始接收里程计和激光雷达消息...")

    # 2. 初始化回调函数的最后打印时间（在init_node之后！）
    odom_callback.last_print_time = rospy.Time.now()
    scan_callback.last_print_time = rospy.Time.now()

    # 3. 订阅话题
    rospy.Subscriber("/odom", Odometry, odom_callback, queue_size=10)
    rospy.Subscriber("/scan", LaserScan, scan_callback, queue_size=10)

    # 4. 保持节点运行
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("传感器监听节点被中断，退出程序。")
