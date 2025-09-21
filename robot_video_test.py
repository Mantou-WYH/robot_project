from robomaster import robot
import numpy as np
import cv2



if __name__ == '__main__':
    robot = robot.Robot()
    robot.initialize(conn_type="ap")
    robot.set_robot_mode(mode='chassis_lead')
    camera = robot.camera
    chassis = robot.chassis
    gimbal = robot.gimbal
    vision = robot.vision

    while True:
        img = camera.read_cv2_image()
        cv2.imshow('Binary Image', img)