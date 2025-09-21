from robomaster import robot
from robomaster import camera
from ultralytics import YOLO
import cv2


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    model = YOLO("/home/scy/robot_project/yolov5s.pt")
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    while True:
        img = ep_camera.read_cv2_image()
        results = model(img)

        # Visualize the results
        img_result = results[0].plot()
        cv2.imshow("test",img_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
