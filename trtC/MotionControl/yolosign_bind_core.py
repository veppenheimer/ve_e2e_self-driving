#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@文件        : yolo_sign_publisher.py
@说明        : ROS 节点，读取 DeepStream-Yolo 识别结果文件，将交通标志类别和面积通过话题发布，
               对同一类别 5 秒内只发布一次。
@作者        : generated
@日期        : 2025-04-24
"""

import rospy
from yolo_msgs.msg import YoloSign  # 需在 your_package/msg/YoloSign.msg 中定义：
# string sign_type
# float32 area
# time stamp
import ctypes, os

# ---------- 绑定函数（CPU 亲和度） ----------
libc = ctypes.CDLL('libc.so.6', use_errno=True)
class cpu_set_t(ctypes.Structure):
    _fields_ = [('__bits', ctypes.c_ulong * 16)]
def CPU_ZERO(setp):
    for i in range(len(setp.__bits)):
        setp.__bits[i] = 0
def CPU_SET(cpu, setp):
    setp.__bits[cpu // 64] |= (1 << (cpu % 64))
def bind_current_thread(cpus):
    """
    将当前线程绑定到指定的 cpu 核心列表上，传入示例：[3]
    """
    mask = cpu_set_t()
    CPU_ZERO(mask)
    for c in cpus:
        CPU_SET(c, mask)
    tid = ctypes.CDLL('libc.so.6').pthread_self()
    res = libc.pthread_setaffinity_np(
        ctypes.c_ulong(tid),
        ctypes.sizeof(mask),
        ctypes.byref(mask)
    )
    if res != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "pthread_setaffinity_np failed: " + os.strerror(errno))
# --------------------------------------------

def main():
    # 绑定主线程到核心 3
    bind_current_thread([3])

    rospy.init_node('yolo_sign_publisher', anonymous=True)
    pub = rospy.Publisher('/yolo_sign', YoloSign, queue_size=10)
    rate = rospy.Rate(20)  # 100 Hz

    # 路径根据实际部署环境调整
    file_path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt'
    # 记录各类别上次发布时刻
    last_pub_time = {}
    filter_duration = rospy.Duration(5.0)   # 其他类别间隔 5 秒
    filter_duration1 = rospy.Duration(0.5)  # 玩偶检测间隔 0.5 秒
    rospy.loginfo('YOLO 识别发布节点启动，监听文件: %s', file_path)

    while not rospy.is_shutdown():
        try:
            with open(file_path, 'r') as f:
                sign_line = f.readline().strip()
                area_line = f.readline().strip()
        except Exception as e:
            rospy.logwarn('读取 YOLO 文件失败: %s', e)
            rate.sleep()
            continue

        # 仅当读取到有效内容时才考虑发布
        if sign_line and area_line:
            try:
                area = float(area_line)
                sign_type = sign_line
                now = rospy.Time.now()
                if sign_type == '0':
                    if area > 2000:  # 4700→0.2
                        sign_type = 'crossing_walk'
                elif sign_type == '1':
                    if area > 10:    # 3500→0.2
                        sign_type = 'limit'
                elif sign_type == '2':
                    sign_type = 'remove_limit'
                elif sign_type == '3':
                    if area > 270:
                        sign_type = 'red'
                elif sign_type == '4':
                    sign_type = 'green'
                elif sign_type == '5':
                    sign_type = 'person'
                else:
                    pass
            except ValueError:
                rospy.logwarn('YOLO 文件内容格式错误: sign="%s", area="%s"', sign_line, area_line)
                rate.sleep()
                continue

            last_time = last_pub_time.get(sign_type)
            # 玩偶 0.5s 内发布一次
            if sign_type == 'person':
                if last_time is None or (now - last_time) > filter_duration1:
                    msg = YoloSign()
                    msg.sign_type = sign_type
                    msg.area = area
                    msg.stamp = now
                    pub.publish(msg)
                    last_pub_time[sign_type] = now
                    rospy.loginfo('发布 YOLO 标志: %s, 面积: %.2f', sign_type, area)
            # 其它类别 5 秒内只发布一次
            else:
                if last_time is None or (now - last_time) > filter_duration:
                    msg = YoloSign()
                    msg.sign_type = sign_type
                    msg.area = area
                    msg.stamp = now
                    pub.publish(msg)
                    last_pub_time[sign_type] = now
                    rospy.loginfo('发布 YOLO 标志: %s, 面积: %.2f', sign_type, area)

        rate.sleep()

    rospy.loginfo('YOLO 发布节点退出')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
