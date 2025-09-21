from robomaster import robot
from robomaster import camera
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync



import cv2


def init_yolo(model_path, conf_thresh=0.5, iou_thresh=0.4):
    """
    初始化 Ultralytics YOLO 模型。

    Parameters:
        model_path (str): YOLO 模型文件的路径。支持 .pt (PyTorch)、.yaml (配置文件) 格式，
                          或者是指向 Ultralytics HUB 模型的名称。
        conf_thresh (float): 置信度阈值。取值范围 [0,1]，默认 0.5。
                            低于此阈值的检测框将被过滤掉。值越高，检测框越少但更可信。
        iou_thresh (float): 非极大值抑制 (NMS) 的 IoU 阈值。取值范围 [0,1]，默认 0.4。
                           用于合并重叠的检测框。值越低，保留的独立检测框越多。

    Returns:
        tuple: 一个包含以下元素的元组：
            - model (ultralytics.YOLO): 加载并初始化好的 YOLO 模型对象。
            - conf_thresh (float): 传入的置信度阈值，方便后续使用。
            - iou_thresh (float): 传入的 IoU 阈值，方便后续使用。

    Raises:
        FileNotFoundError: 如果指定的 model_path 不存在。
        ValueError: 如果模型文件格式不支持或参数值无效。
    """
    # 加载模型
    model = YOLO(model_path)
    print(f"YOLO model '{model_path}' loaded successfully!")
    return model, conf_thresh, iou_thresh


def yolo_detect_image(model, conf_thresh, iou_thresh, image):
    """
    使用初始化的 Ultralytics YOLO 模型检测单张图片中的物体。

    Parameters:
        model (ultralytics.YOLO): 通过 init_yolo 函数初始化好的 YOLO 模型对象。
        conf_thresh (float): 置信度阈值。
        iou_thresh (float): 非极大值抑制 (NMS) 的 IoU 阈值。
        image (numpy.ndarray): 输入图像，通常是一个通过 OpenCV 读取的 BGR 格式的 NumPy 数组。

    Returns:
        numpy.ndarray: 一个绘制了检测框、标签和置信度的图像（BGR格式）。
    """
    # 使用模型进行预测
    results = model.predict(
        source=image,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False  # 设置为 True 会在控制台输出详细检测信息
    )

    # 获取检测结果并绘制到图像上
    result = results[0]  # 因为只预测了一张图片，所以取第一个结果
    annotated_frame = result.plot()  # 这个函数返回绘制好的图像（BGR格式）

    return annotated_frame

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    model, conf_thresh, iou_thresh = init_yolo(
        model_path="yolov8s.pt", # 替换成你的模型路径
        conf_thresh=0.5,
        iou_thresh=0.4
    )
    while True:
        img = ep_camera.read_cv2_image()
        if img is None:
            print("Error: Could not load image")
        else:
            # 3. 进行目标检测
            result_img = yolo_detect_image(model, conf_thresh, iou_thresh, img)

            # 4. 显示和保存结果
            cv2.imshow("YOLO Detection Result", result_img)
