#from robomaster import robot
#from robomaster import camera
from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    #ep_robot = robot.Robot()
    #ep_robot.initialize(conn_type="ap")
    #ep_camera = ep_robot.camera
    # Load a pretrained YOLO model
    model = YOLO("/home/scy/robot_project/yolov5s.pt")

    # Perform object detection on an image
    #img = ep_camera.read_cv2_image()
    results = model("/home/scy/robot_project/detal/test1.jpeg")

    # Visualize the results
    img = results[0].plot()
    cv2.imshow("test",img)
    cv2.waitKey(3000)
