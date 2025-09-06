#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

def mybot_velocity_control():
    # 1. 初始化ROS节点（节点名：mybot_vel_controller）
    rospy.init_node('mybot_vel_controller', anonymous=True)
    
    # 2. 创建速度指令发布者（话题：cmd_vel，与你的机器人插件配置一致）
    vel_pub = rospy.Publisher(
        "/cmd_vel",  # 你的机器人实际监听的速度指令话题
        Twist, 
        queue_size=10  # 消息队列大小，防止指令丢失
    )
    
    # 3. 等待发布者与机器人底盘（diff_drive插件）建立连接
    rospy.loginfo("等待与 mybot 底盘建立连接...")
    while vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.5)  # 每0.5秒检查一次连接状态
    rospy.loginfo("已与 mybot 底盘建立连接，开始发送速度指令！")
    
    # 4. 定义速度指令（Twist消息：linear.x=线速度，angular.z=角速度）
    vel_msg = Twist()
    rate = rospy.Rate(10)  # 控制频率：10Hz（确保底盘稳定接收指令）
    
    # 5. 运动逻辑：前进→左转→前进→右转→停止（可按需修改速度/时间）
    try:
        # 阶段1：前进（线速度0.2m/s，无旋转）
        rospy.loginfo("阶段1：前进（2秒）")
        vel_msg.linear.x = 0.2  # 线速度（正值前进，负值后退；建议0.1~0.3，避免不稳定）
        vel_msg.angular.z = 0.0 # 角速度（0=不旋转）
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
            vel_pub.publish(vel_msg)
            rate.sleep()
        
        # 阶段2：左转（原地旋转，角速度0.5rad/s）
        rospy.loginfo("阶段2：左转（2秒）")
        vel_msg.linear.x = 0.0  # 停止前进
        vel_msg.angular.z = 0.5 # 角速度（正值左转，负值右转；建议0.3~0.8）
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
            vel_pub.publish(vel_msg)
            rate.sleep()
        
        # 阶段3：前进（再次前进）
        rospy.loginfo("阶段3：前进（2秒）")
        vel_msg.linear.x = 0.2
        vel_msg.angular.z = 0.0
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
            vel_pub.publish(vel_msg)
            rate.sleep()
        
        # 阶段4：右转（原地旋转）
        rospy.loginfo("阶段4：右转（2秒）")
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = -0.5 # 负值=右转
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
            vel_pub.publish(vel_msg)
            rate.sleep()
        
        # 阶段5：停止运动（发送0速度，确保机器人停稳）
        rospy.loginfo("阶段5：停止运动")
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        for _ in range(5):  # 连续发送5次停止指令，避免指令丢失
            vel_pub.publish(vel_msg)
            rate.sleep()
        
        rospy.loginfo("所有运动阶段完成！")

    except rospy.ROSInterruptException:
        # 捕获Ctrl+C等中断信号，立即停止机器人
        rospy.loginfo("运动被中断，立即停止 mybot！")
        vel_msg.linear.x = 0.0
        vel_msg.angular.z = 0.0
        vel_pub.publish(vel_msg)

if __name__ == "__main__":
    # 直接启动控制逻辑（无需机器人编号参数）
    mybot_velocity_control()