# #!/usr/bin/env python
# import rospy
# from geometry_msgs.msg import Twist
# 
# def mybot_velocity_control():
#     # 1. 初始化ROS节点（节点名：mybot_vel_controller）
#     rospy.init_node('mybot_vel_controller', anonymous=True)
#     
#     # 2. 创建速度指令发布者（话题：cmd_vel，与你的机器人插件配置一致）
#     vel_pub = rospy.Publisher(
#         "/cmd_vel",  # 你的机器人实际监听的速度指令话题
#         Twist, 
#         queue_size=10  # 消息队列大小，防止指令丢失
#     )
#     
#     # 3. 等待发布者与机器人底盘（diff_drive插件）建立连接
#     rospy.loginfo("等待与 mybot 底盘建立连接...")
#     while vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
#         rospy.sleep(0.5)  # 每0.5秒检查一次连接状态
#     rospy.loginfo("已与 mybot 底盘建立连接，开始发送速度指令！")
#     
#     # 4. 定义速度指令（Twist消息：linear.x=线速度，angular.z=角速度）
#     vel_msg = Twist()
#     rate = rospy.Rate(10)  # 控制频率：10Hz（确保底盘稳定接收指令）
#     
#     # 5. 运动逻辑：前进→左转→前进→右转→停止（可按需修改速度/时间）
#     try:
#         # 阶段1：前进（线速度0.2m/s，无旋转）
#         rospy.loginfo("阶段1：前进（2秒）")
#         vel_msg.linear.x = 0.2  # 线速度（正值前进，负值后退；建议0.1~0.3，避免不稳定）
#         vel_msg.angular.z = 0.0 # 角速度（0=不旋转）
#         start_time = rospy.Time.now()
#         while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
#             vel_pub.publish(vel_msg)
#             rate.sleep()
#         
#         # 阶段2：左转（原地旋转，角速度0.5rad/s）
#         rospy.loginfo("阶段2：左转（2秒）")
#         vel_msg.linear.x = 0.0  # 停止前进
#         vel_msg.angular.z = 0.5 # 角速度（正值左转，负值右转；建议0.3~0.8）
#         start_time = rospy.Time.now()
#         while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
#             vel_pub.publish(vel_msg)
#             rate.sleep()
#         
#         # 阶段3：前进（再次前进）
#         rospy.loginfo("阶段3：前进（2秒）")
#         vel_msg.linear.x = 0.2
#         vel_msg.angular.z = 0.0
#         start_time = rospy.Time.now()
#         while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
#             vel_pub.publish(vel_msg)
#             rate.sleep()
#         
#         # 阶段4：右转（原地旋转）
#         rospy.loginfo("阶段4：右转（2秒）")
#         vel_msg.linear.x = 0.0
#         vel_msg.angular.z = -0.5 # 负值=右转
#         start_time = rospy.Time.now()
#         while (rospy.Time.now() - start_time).to_sec() < 2.0 and not rospy.is_shutdown():
#             vel_pub.publish(vel_msg)
#             rate.sleep()
#         
#         # 阶段5：停止运动（发送0速度，确保机器人停稳）
#         rospy.loginfo("阶段5：停止运动")
#         vel_msg.linear.x = 0.0
#         vel_msg.angular.z = 0.0
#         for _ in range(5):  # 连续发送5次停止指令，避免指令丢失
#             vel_pub.publish(vel_msg)
#             rate.sleep()
#         
#         rospy.loginfo("所有运动阶段完成！")
# 
#     except rospy.ROSInterruptException:
#         # 捕获Ctrl+C等中断信号，立即停止机器人
#         rospy.loginfo("运动被中断，立即停止 mybot！")
#         vel_msg.linear.x = 0.0
#         vel_msg.angular.z = 0.0
#         vel_pub.publish(vel_msg)
# 
# if __name__ == "__main__":
#     # 直接启动控制逻辑（无需机器人编号参数）
#     mybot_velocity_control()

#!/usr/bin/env python
import rospy
import time
import tf.transformations  # 用于四元数转欧拉角
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

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
    rospy.loginfo("已与 mybot 底盘建立连接，准备测试初始状态！")
    
    # 4. 测试初始状态：获取并显示初始位姿参数（含欧拉角）
    rospy.loginfo("===== 开始测试初始状态 =====")
    init_state = {
        "x_init": None,
        "y_init": None,
        "z_init": None,
        "x_rot_init": None,
        "y_rot_init": None,
        "z_rot_init": None,
        "w_rot_init": None,
        "roll_init": None,   # 横滚角（绕x轴旋转）
        "pitch_init": None,  # 俯仰角（绕y轴旋转）
        "yaw_init": None     # 偏航角（绕z轴旋转，转向角）
    }
    
    # 4.1 尝试从参数服务器获取初始参数（如果有预设）
    try:
        init_state["x_init"] = rospy.get_param("/mybot/initial_pose/x_init")
        init_state["y_init"] = rospy.get_param("/mybot/initial_pose/y_init")
        init_state["z_init"] = rospy.get_param("/mybot/initial_pose/z_init", 0.0)
        init_state["x_rot_init"] = rospy.get_param("/mybot/initial_pose/x_rot_init")
        init_state["y_rot_init"] = rospy.get_param("/mybot/initial_pose/y_rot_init")
        init_state["z_rot_init"] = rospy.get_param("/mybot/initial_pose/z_rot_init")
        init_state["w_rot_init"] = rospy.get_param("/mybot/initial_pose/w_rot_init")
        rospy.loginfo("从参数服务器获取初始状态成功！")
        
        # 从四元数计算欧拉角（参数服务器获取的四元数）
        quaternion = [
            init_state["x_rot_init"],
            init_state["y_rot_init"],
            init_state["z_rot_init"],
            init_state["w_rot_init"]
        ]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        init_state["roll_init"] = roll
        init_state["pitch_init"] = pitch
        init_state["yaw_init"] = yaw
        
    except:
        rospy.loginfo("参数服务器未找到初始参数，尝试从里程计获取...")
        
        # 4.2 从里程计话题（/odom）获取初始位姿
        odom_data = None
        def odom_callback(msg):
            nonlocal odom_data
            odom_data = msg
        
        odom_sub = rospy.Subscriber("/odom", Odometry, odom_callback)
        
        # 等待里程计数据（超时5秒）
        timeout = 5
        start_time = time.time()
        while odom_data is None and (time.time() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if odom_data is not None:
            # 提取位置（x, y, z）
            init_state["x_init"] = odom_data.pose.pose.position.x
            init_state["y_init"] = odom_data.pose.pose.position.y
            init_state["z_init"] = odom_data.pose.pose.position.z
            # 提取旋转四元数（x, y, z, w）
            init_state["x_rot_init"] = odom_data.pose.pose.orientation.x
            init_state["y_rot_init"] = odom_data.pose.pose.orientation.y
            init_state["z_rot_init"] = odom_data.pose.pose.orientation.z
            init_state["w_rot_init"] = odom_data.pose.pose.orientation.w
            
            # 四元数转欧拉角（单位：弧度）
            quaternion = [
                init_state["x_rot_init"],
                init_state["y_rot_init"],
                init_state["z_rot_init"],
                init_state["w_rot_init"]
            ]
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
            init_state["roll_init"] = roll
            init_state["pitch_init"] = pitch
            init_state["yaw_init"] = yaw
            
            rospy.loginfo("从里程计获取初始状态成功！")
        else:
            rospy.logwarn("获取初始状态超时！使用默认值。")
            # 若获取失败，使用默认值（无旋转状态）
            init_state = {
                "x_init": 0.0,
                "y_init": 0.0,
                "z_init": 0.0,
                "x_rot_init": 0.0,
                "y_rot_init": 0.0,
                "z_rot_init": 0.0,
                "w_rot_init": 1.0,
                "roll_init": 0.0,
                "pitch_init": 0.0,
                "yaw_init": 0.0
            }
    
    # 4.3 显示初始状态参数（欧拉角同时显示弧度和角度，方便阅读）
    rospy.loginfo("初始状态参数：")
    for key, value in init_state.items():
        if key in ["roll_init", "pitch_init", "yaw_init"]:
            # 转换为角度（弧度 * 180/π）
            degrees = round(value * 180 / 3.14159, 4)
            rospy.loginfo(f"  {key}: {round(value, 4)} 弧度 ({degrees} 度)")
        else:
            rospy.loginfo(f"  {key}: {round(value, 4)}")
    rospy.loginfo("===== 初始状态测试完成 =====")
    rospy.sleep(2)  # 暂停2秒，方便查看结果
    
    # 5. 定义速度指令（Twist消息：linear.x=线速度，angular.z=角速度）
    vel_msg = Twist()
    rate = rospy.Rate(10)  # 控制频率：10Hz（确保底盘稳定接收指令）
    
    # 6. 运动逻辑：前进→左转→前进→右转→停止（可按需修改速度/时间）
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
#!/usr/bin/env python
import rospy
import time
import tf.transformations  # 用于四元数转欧拉角
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

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
    rospy.loginfo("已与 mybot 底盘建立连接，准备测试初始状态！")
    
    # 4. 测试初始状态：获取并显示初始位姿参数（含欧拉角）
    rospy.loginfo("===== 开始测试初始状态 =====")
    init_state = {
        "x_init": None,
        "y_init": None,
        "z_init": None,
        "x_rot_init": None,
        "y_rot_init": None,
        "z_rot_init": None,
        "w_rot_init": None,
        "roll_init": None,   # 横滚角（绕x轴旋转）
        "pitch_init": None,  # 俯仰角（绕y轴旋转）
        "yaw_init": None     # 偏航角（绕z轴旋转，转向角）
    }
    
    # 4.1 尝试从参数服务器获取初始参数（如果有预设）
    try:
        init_state["x_init"] = rospy.get_param("/mybot/initial_pose/x_init")
        init_state["y_init"] = rospy.get_param("/mybot/initial_pose/y_init")
        init_state["z_init"] = rospy.get_param("/mybot/initial_pose/z_init", 0.0)
        init_state["x_rot_init"] = rospy.get_param("/mybot/initial_pose/x_rot_init")
        init_state["y_rot_init"] = rospy.get_param("/mybot/initial_pose/y_rot_init")
        init_state["z_rot_init"] = rospy.get_param("/mybot/initial_pose/z_rot_init")
        init_state["w_rot_init"] = rospy.get_param("/mybot/initial_pose/w_rot_init")
        rospy.loginfo("从参数服务器获取初始状态成功！")
        
        # 从四元数计算欧拉角（参数服务器获取的四元数）
        quaternion = [
            init_state["x_rot_init"],
            init_state["y_rot_init"],
            init_state["z_rot_init"],
            init_state["w_rot_init"]
        ]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        init_state["roll_init"] = roll
        init_state["pitch_init"] = pitch
        init_state["yaw_init"] = yaw
        
    except:
        rospy.loginfo("参数服务器未找到初始参数，尝试从里程计获取...")
        
        # 4.2 从里程计话题（/odom）获取初始位姿
        odom_data = None
        def odom_callback(msg):
            nonlocal odom_data
            odom_data = msg
        
        odom_sub = rospy.Subscriber("/odom", Odometry, odom_callback)
        
        # 等待里程计数据（超时5秒）
        timeout = 5
        start_time = time.time()
        while odom_data is None and (time.time() - start_time) < timeout:
            rospy.sleep(0.1)
        
        if odom_data is not None:
            # 提取位置（x, y, z）
            init_state["x_init"] = odom_data.pose.pose.position.x
            init_state["y_init"] = odom_data.pose.pose.position.y
            init_state["z_init"] = odom_data.pose.pose.position.z
            # 提取旋转四元数（x, y, z, w）
            init_state["x_rot_init"] = odom_data.pose.pose.orientation.x
            init_state["y_rot_init"] = odom_data.pose.pose.orientation.y
            init_state["z_rot_init"] = odom_data.pose.pose.orientation.z
            init_state["w_rot_init"] = odom_data.pose.pose.orientation.w
            
            # 四元数转欧拉角（单位：弧度）
            quaternion = [
                init_state["x_rot_init"],
                init_state["y_rot_init"],
                init_state["z_rot_init"],
                init_state["w_rot_init"]
            ]
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
            init_state["roll_init"] = roll
            init_state["pitch_init"] = pitch
            init_state["yaw_init"] = yaw
            
            rospy.loginfo("从里程计获取初始状态成功！")
        else:
            rospy.logwarn("获取初始状态超时！使用默认值。")
            # 若获取失败，使用默认值（无旋转状态）
            init_state = {
                "x_init": 0.0,
                "y_init": 0.0,
                "z_init": 0.0,
                "x_rot_init": 0.0,
                "y_rot_init": 0.0,
                "z_rot_init": 0.0,
                "w_rot_init": 1.0,
                "roll_init": 0.0,
                "pitch_init": 0.0,
                "yaw_init": 0.0
            }
    
    # 4.3 显示初始状态参数（欧拉角同时显示弧度和角度，方便阅读）
    rospy.loginfo("初始状态参数：")
    for key, value in init_state.items():
        if key in ["roll_init", "pitch_init", "yaw_init"]:
            # 转换为角度（弧度 * 180/π）
            degrees = round(value * 180 / 3.14159, 4)
            rospy.loginfo(f"  {key}: {round(value, 4)} 弧度 ({degrees} 度)")
        else:
            rospy.loginfo(f"  {key}: {round(value, 4)}")
    rospy.loginfo("===== 初始状态测试完成 =====")
    rospy.sleep(2)  # 暂停2秒，方便查看结果
    
    # 5. 定义速度指令（Twist消息：linear.x=线速度，angular.z=角速度）
    vel_msg = Twist()
    rate = rospy.Rate(10)  # 控制频率：10Hz（确保底盘稳定接收指令）
    
    # 6. 运动逻辑：前进→左转→前进→右转→停止（可按需修改速度/时间）
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
