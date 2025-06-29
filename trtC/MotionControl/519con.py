#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from yolo_msgs.msg import YoloSign
import numpy as np
import cv2
import time

# 导入 TensorRT 相关库
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 驱动

from collections import deque

class MotionController:

    def __init__(self):
        # 参数：TensorRT 引擎路径和摄像头索引
        engine_path = rospy.get_param('~engine_path', 'results/ve_0512_100b_simplified_fp16.engine')
        camera_index = rospy.get_param('~camera_index', 0)

        # 初始化 ROS 发布/订阅
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/yolo_sign', YoloSign, self.sign_callback)

        # 打开摄像头
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            rospy.logerr("无法打开摄像头（索引 %d）", camera_index)
            rospy.signal_shutdown("摄像头无法打开")

        # 交通标志队列与当前/上一个动作状态
        self.sign_queue = deque()
        self.current_action = None
        self.current_sign = None
        self.prev_processed_sign = None
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.limit_handled = False

        self.delay_before_red = 7.2

        # 加载 TensorRT 引擎
        rospy.loginfo("Loading TensorRT engine from %s ...", engine_path)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        rospy.loginfo("TensorRT engine loaded.")
        self.context = self.engine.create_execution_context()

        # 引擎绑定信息
        input_idx = self.engine.get_binding_index("input")
        output_idx = self.engine.get_binding_index("output")
        INPUT_SHAPE = (1, 3, 120, 160)
        input_size = int(np.prod(INPUT_SHAPE) * np.float32().itemsize)
        output_size = int(np.prod((1,)) * np.float32().itemsize)
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.stream = cuda.Stream()

        # 控制参数
        self.default_linear_speed = 0.2
        self.slow_linear_speed = 0.12
        self.turn_angular_speed = 0.5
        self.stop1_duration = 2.0
        self.stop2_duration = 100.0
        self.stop_duration_person = 1.0
        self.slow_duration = 40.0
        self.turn_duration = 3.0

        # 定时循环 10Hz
        self.rate = rospy.Rate(10)

    def clear_sign_file(self):
        '''清空交通标志文件内容'''
        file_path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt'
        try:
            with open(file_path, 'w'):
                pass
            rospy.loginfo("Cleared sign file: %s", file_path)
        except Exception as e:
            rospy.logerr("Failed to clear sign file %s: %s", file_path, e)

    def sign_callback(self, msg):
        '''YOLO 标志消息回调，加入队列或处理特殊情况'''
        if msg.sign_type == 'limit' and self.limit_handled:
            return
        if getattr(self, 'current_sign', None) == 'red' and msg.sign_type == 'green':
            rospy.loginfo("Green light detected, finishing red_light action")
            self.current_action = None
            self.current_sign = None
            self.prev_processed_sign = msg.sign_type
            self.clear_sign_file()
            return
        if getattr(self, 'current_sign', None) == 'limit' and msg.sign_type == 'remove_limit':
            rospy.loginfo("Speed limit lift detected, finishing speed_limit action")
            self.current_action = None
            self.current_sign = None
            self.prev_processed_sign = msg.sign_type
            self.clear_sign_file()
            return
        rospy.loginfo("Queued sign: %s", msg.sign_type)
        self.sign_queue.append(msg.sign_type)

    def preprocess(self, frame):
        '''图像预处理：resize(160x120)、BGR->HSV、归一化、CHW'''
        img = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        norm = hsv.astype(np.float32) / 255.0
        chw = norm.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0).copy()

    def infer_trt(self, img_np):
        '''执行 TRT 推理，返回角速度'''
        cuda.memcpy_htod_async(self.d_input, img_np, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)],
                                     stream_handle=self.stream.handle)
        output = np.empty((1,), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return float(output[0])

    def spin(self):
        '''主循环：处理队列动作或 TRT 推理，并发布 /cmd_vel'''
        while not rospy.is_shutdown():
            twist = Twist()
            now = rospy.get_time()

            # 如果无当前动作且队列中有待处理标志，则启动下一动作
            if self.current_action is None and self.sign_queue:
                sign = self.sign_queue.popleft()
                rospy.loginfo("Processing queued sign: %s", sign)
                if sign == 'red' and self.prev_processed_sign != 'remove_limit':
                    rospy.loginfo("Skipping red light because previous sign is not 'remove_limit'.")
                    continue
                if sign == 'red':
                    action = 'delay_red'; duration = self.delay_before_red
                elif sign == 'limit':
                    action = 'slow'; duration = self.slow_duration
                    self.limit_handled = True
                elif sign in ('green', 'remove_limit'):
                    continue
                elif sign == 'crossing_walk':
                    action = 'stop1'; duration = self.stop1_duration
                elif sign == 'person':
                    action = 'person'; duration = self.stop_duration_person
                else:
                    rospy.logwarn("Unknown sign %s, skipping", sign)
                    continue

                self.current_action = action
                self.current_sign = sign
                self.action_duration = duration
                self.action_start_time = now
                rospy.loginfo("Start action %s for %.1f s", action, duration)

            # 有当前动作：包括 delay_red, stop1, stop2, slow, turn 等
            if self.current_action:
                elapsed = now - self.action_start_time

                # 红灯预备延时阶段
                if self.current_action == 'delay_red':
                    if elapsed < self.action_duration:
                        # 预备期间正常模型推理行驶
                        ret, frame = self.cap.read()
                        if ret:
                            img_in = self.preprocess(frame)
                            ang_z = self.infer_trt(img_in)
                            twist.linear.x = self.default_linear_speed
                            twist.angular.z = ang_z
                        else:
                            twist.linear.x = twist.angular.z = 0.0
                        self.twist_pub.publish(twist)
                        self.rate.sleep()
                        continue
                    else:
                        # 预备延时结束，进入真正停车
                        self.current_action = 'stop2'
                        self.action_duration = self.stop2_duration
                        self.action_start_time = now
                        continue

                # 停车、慢行、转向、行人停止等逻辑
                if self.current_action == 'stop1' or self.current_action == 'stop2':
                    if elapsed < self.action_duration:
                        twist.linear.x = twist.angular.z = 0.0
                        self.twist_pub.publish(twist)
                        self.rate.sleep()
                        continue
                    else:
                        rospy.loginfo("%s completed", self.current_action)
                        self.clear_sign_file()
                        self.prev_processed_sign = self.current_sign
                        self.current_action = None
                        self.current_sign = None
                        continue
                if self.current_action == 'slow':
                    if elapsed < self.action_duration:
                        ret_frame, frame_slow = self.cap.read()
                        if ret_frame:
                            img_slow = self.preprocess(frame_slow)
                            twist.angular.z = self.infer_trt(img_slow)
                        else:
                            twist.angular.z = 0.0
                        twist.linear.x = self.slow_linear_speed
                        self.twist_pub.publish(twist)
                        self.rate.sleep()
                        continue
                    else:
                        rospy.loginfo("Slow completed")
                        self.clear_sign_file()
                        self.prev_processed_sign = self.current_sign
                        self.current_action = None
                        self.current_sign = None
                        continue

            # 正常 TRT 推理流程
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("读取摄像头失败，跳过本次循环")
                self.rate.sleep()
                continue

            try:
                img_in = self.preprocess(frame)
                ang_z = self.infer_trt(img_in)
                twist.linear.x = self.default_linear_speed
                twist.angular.z = ang_z
                self.twist_pub.publish(twist)
            except Exception as e:
                rospy.logerr("推理或发布失败: %s", e)
                twist.linear.x = twist.angular.z = 0.0
                self.twist_pub.publish(twist)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('motion_controller_node', anonymous=False)
    node = MotionController()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
