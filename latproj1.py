import cv2
from ultralytics import YOLO
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO("yolov8n.pt")
    while True:
        ret,frame=cap.read()
        cv2.imshow("yolov8",frame)
        detection=model(frame)[0]
        if cv2.waitKey(30) == 27 :
            break



if __name__ == "__main__" :
    main()