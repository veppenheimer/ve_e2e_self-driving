#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@文件        :data_collector_motion.py
@说明        :ROS节点，根据预设运动阶段采集数据，同时控制小车运动
@时间        :2022/03/XX
@作者        :你的名字
"""

import rospy
import cv2
import time
import os
from geometry_msgs.msg import Twist

def data_collector():
    # 初始化 ROS 节点
    rospy.init_node('data_collector_motion', anonymous=True)
    
    # 创建发布者，发布到小车控制的话题（假设话题为 /cmd_vel）
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    # 设置发布频率 (例如 10 Hz)
    rate = rospy.Rate(10)
    
    # 打开摄像头（索引 0）
    cap = cv2.VideoCapture("/dev/video0")
    if not cap.isOpened():
        rospy.logerr("无法打开摄像头")
        return

    # 数据采集保存目录
    save_dir = "data_log"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 定义运动阶段列表
    # 每个阶段包含持续时间 (秒)、线速度和转向角（angular.z）
    phases = [


    #stage1
      # {"duration": 5.406, "linear": 0.2, "angular": 0.0},
      # {"duration": 1.81, "linear":  0.2, "angular": 0.6827},
      # {"duration": 0.052, "linear": 0.2, "angular": 0.0},
      # {"duration": 5.402, "linear": 0.2, "angular": 0.6655},
    # #new stage1
    #    {"duration": 1.71, "linear": 0.5, "angular": 0.0},
    #    {"duration": 0.6, "linear":  0.5, "angular": 1.8},
    #    {"duration": 0.056, "linear": 0.5, "angular": 0.0},
    #    {"duration": 2.248, "linear": 0.5, "angular": 1.55},
    #    {"duration": 1.080, "linear": 0.5, "angular": 0.0},
      
    #stage2          
      # {"duration": 2.205, "linear": 0.2, "angular": 0.0},
      # {"duration": 3.335, "linear": 0.2, "angular": -0.7650},
      # {"duration": 3.08, "linear": 0.2, "angular": 0.0},
      # {"duration": 5.3, "linear": 0.2, "angular": -0.7292},
      # {"duration": 0.45, "linear": 0.2, "angular": 0.0},
      # {"duration": 4.977, "linear": 0.2, "angular": -0.6945},
    #  newStage 2
        {"duration": 0.912, "linear": 0.5, "angular": 0.0},
        {"duration": 1.137, "linear": 0.5, "angular": -1.913},
        {"duration": 1.286, "linear": 0.5, "angular": 0.0},
        {"duration": 2.030, "linear": 0.5, "angular": -1.563},
        {"duration": 0.160, "linear": 0.5, "angular": 0.0},
        {"duration": 1.762, "linear": 0.5, "angular": -1.636},
      

     # Newstage3
     #   {"duration": 3.0, "linear": 0.2, "angular": 0.0},

        # Newstage3
      # {"duration": 1.2, "linear": 0.5, "angular": 0.0},
     #
     # New#stage4
     #      {"duration": 0.385, "linear": 0.5, "angular": 0.0},
     #      {"duration": 0.603, "linear":  0.5, "angular": -1.78},
     #      {"duration": 0.502, "linear": 0.5, "angular":  0.0},
     #      {"duration": 0.603, "linear":  0.5, "angular": 1.78},
     #
     #      {"duration": 0.385, "linear":  0.5, "angular": 0.0},
     #      {"duration": 0.595, "linear": 0.5, "angular": 1.78},
     #      {"duration": 0.72, "linear": 0.5, "angular":  0.0},
     #      {"duration": 0.573, "linear": 0.5, "angular": -1.78},
#New
         #   {"duration": 1.2, "linear": 0.5, "angular": 0.0},
          #  {"duration": 1.65, "linear": 0.0, "angular": 1.5},

           # {"duration": 0.35, "linear": 0.5, "angular": 0.0},
            #{"duration": 2.30, "linear": 0.5, "angular": -1.5},
            #{"duration": 0.25, "linear": 0.5, "angular": -1.35},
            #{"duration": 0.476, "linear": 0.5, "angular": 0.0},
            #{"duration": 0.62, "linear": 0.5, "angular": 2.15},
            
            #{"duration": 0.55, "linear": 0.5, "angular": 0.0},


       #stage5
       # {"duration": 1.15, "linear": 0.2, "angular": 0.0},
       # {"duration": 3.05, "linear": 0.2, "angular": 0.75},

       # {"duration": 1.5, "linear": 0.2, "angular": 0.0}

     
    ]
    
    # 初始化 Twist 消息
    twist_msg = Twist()
    
    pic_index = 0  # 图像计数器
    
    rospy.loginfo("开始数据采集")
    
    # 对每个运动阶段执行相应动作并采集数据
    for phase in phases:
        phase_start = time.time()
        while not rospy.is_shutdown() and (time.time() - phase_start < phase["duration"]):
            # 设置当前阶段控制指令
            twist_msg.linear.x = phase["linear"]
            twist_msg.angular.z = phase["angular"]
            cmd_pub.publish(twist_msg)
            
            # 采集一帧图像
            ret, frame = cap.read()
            if not ret:
                rospy.logerr("采集图像失败")
                break
            
            # 保存图像：文件名格式为 "log/{pic_index}_{steering_angle:.4f}.jpg"
            save_path = os.path.join(save_dir, "{:d}_{:.4f}.jpg".format(pic_index, twist_msg.angular.z))
            cv2.imwrite(save_path, frame)
            rospy.loginfo("保存图像: " + save_path)
            
            pic_index += 1
            rate.sleep()
    
    # 运动结束后，发送停止命令
    twist_msg.linear.x = 0.0
    twist_msg.angular.z = 0.0
    cmd_pub.publish(twist_msg)
    
    # 关闭摄像头
    cap.release()
    rospy.loginfo("数据采集结束，摄像头已关闭.")

if __name__ == '__main__':
    try:
        data_collector()
    except rospy.ROSInterruptException:
        pass
