#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import cv2
import numpy as np
import math
from time import time, strftime, localtime
import zmq
import base64


def detect_edges(frame):
    '''
    检测蓝色区域边缘
    '''
    # 高斯滤波滤除小的噪点
    frame = cv2.GaussianBlur(frame,(9,9),2)
    # 特定颜色区域提取
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([50, 30, 30])
    upper_blue = np.array([100, 255, 255])
    #lower_blue = np.array([40, 40, 40])
    #upper_blue = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = region_of_interest(mask)

    # 连通域分析
    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_new = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        w,h=rect[1]
            
        # if max(w,h)<3*min(w,h): # 长宽比不合适
        #     contours_new.append(c)
        #     continue
        if w*h < 500: # 面积太小
            contours_new.append(c)
            continue

    mask = cv2.fillPoly(mask, contours_new, (0,))
    #cv2.imwrite("mask.jpg", mask)
    edges = cv2.Canny(mask, 200, 400)
    return edges


def region_of_interest(edges):
    '''
    提取感兴趣区域（截取下半部分）
    '''
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges


def detect_line_segments(cropped_edges):
    '''
    霍夫变换检测
    '''
    rho = 1  # 距离精度, 以像素为单位
    angle = np.pi / 180  # 径向角度精度, 以度为单位
    min_threshold = 10  # 最小投票数
    line_segments = cv2.HoughLinesP(cropped_edges,
                                    rho,
                                    angle,
                                    min_threshold,
                                    np.array([]),
                                    minLineLength=8,
                                    maxLineGap=4)

    return line_segments


def make_points(frame, line):
    '''
    根据直线斜率和截距返回对应的线段两端坐标
    '''
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 1 / 2)

    # 限制坐标在图像区域内
    x1 = max(-width, min(2 * width, int((y1 - intercept) / (slope+0.000001))))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / (slope+0.000001))))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    汇聚所有线段成1段或2段
    如果所有线段斜率  slopes < 0: 只检测到左边行道线
    如果所有线段斜率  slopes > 0: 只检测到右边行道线
    """
    lane_lines = []
    if line_segments is None:
        print('没有检测到线段')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # 左行道线应该位于整个图像的左2/3部分
    right_region_boundary = width * boundary  # 右行道线应该位于整个图像的右2/3部分

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:  # 忽略垂直线（没有斜率）
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -math.tan(25):
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            elif slope > math.tan(25):
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
   
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_points(frame, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)

        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines


def detect_lane(frame):
    '''
    检测线段
    '''
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)

    return lane_lines


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    '''
    对检测到的线段进行可视化展示
    '''
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color,
                         line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def compute_steer(lane_lines, frame):
    height, width, _ = frame.shape
    x_offset = 0
    y_offset = 0
    line_num = 0
    steering_angle = 0
    if len(lane_lines) == 2:  # 检测到2条行道线
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)
        line_num = 2
        print('检测到2条线')
    elif len(lane_lines) == 1:  # 检测到1条行道线
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = int((x2 - x1)/1.0)
        y_offset = int(height / 2.0)
        line_num = 1
        print('检测到1条线')
    else:
        print('检测失败')
        return 0, 0

    angle_to_mid_radian = math.atan(
        x_offset / y_offset)  # angle (in radian) to center vertical line
    steering_angle = int(angle_to_mid_radian * 180.0 /
                         math.pi)  # angle (in degrees) to center vertical line
    return line_num, steering_angle


def stabilize_steering_angle(curr_steering_angle,
                             new_steering_angle,
                             num_of_lane_lines,
                             max_angle_deviation_two_lines=5,
                             max_angle_deviation_one_lane=3):
    """
    用于平稳控制小车转向
    如果当前计算出来的转向角与上一帧差距太大，则做幅度限制
    """
    new_steering_angle = int(new_steering_angle * 1.0 / 1)  # 降低转向灵敏度

    if num_of_lane_lines == 2:
        # 检测到2条线，我们用更快得调整幅度
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        # 检测到1条线，我们缩小调整幅度
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle +
                                        max_angle_deviation * angle_deviation /
                                        abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle

    # 限定转向值上限
    if stabilized_steering_angle > 35:
        stabilized_steering_angle = 35
    if stabilized_steering_angle < -35:
        stabilized_steering_angle = -35
    return stabilized_steering_angle


def take_photo(steer_angle, frame, pic_index):
    '''
    采集照片和对应的转向值
    '''
    _time = strftime('%Y-%m-%d-%H-%M-%S', localtime(time()))
    name = '%s' % _time
    img_path = "./log/" + name + '_photo' + str(pic_index) + '_' + str(steer_angle) + '.jpg'
    cv2.imwrite(img_path, frame)


class ImageTrans(object):
    '''
    视频图像传输
    '''
    def __init__(self, ip):
        self.contest = zmq.Context()
        self.footage_socket = self.contest.socket(zmq.PAIR)
        self.footage_socket.connect('tcp://%s:5555'%ip)
    
    def sendImg(self,img):
        '''
        发送图像给上位机
        '''
        # 转换为流数据并编码
        encoded, buffer = cv2.imencode('.jpg', img) 
        jpg_as_test = base64.b64encode(buffer) #把内存中的图像流数据进行base64编码

        # 发送数据
        self.footage_socket.send(jpg_as_test) #把编码后的流数据发送给视频的接收端

