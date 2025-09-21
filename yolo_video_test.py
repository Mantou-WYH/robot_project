from robomaster import robot
from robomaster import camera
from ultralytics import YOLO



if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    # Load a pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Perform object detection on an image
    img = ep_camera.read_cv2_image()
    results = model(img)

    # Visualize the results
    for result in results:
        result.show()
