import cv2
from ultralytics import YOLO
import supervision as sv
from argparse import ArgumentParser

def args():
    parser = ArgumentParser(description='LOTICv3')
    parser.add_argument('video_source', type=str, help='identify video source')
    parser.add_argument('model', type=str, help='path/to/yolov8/model')
    parser.add_argument('conf', type=float, default=0.25 ,help='detection confidence threshold')

def main():
    #setup global variables for main detection loop
    model = YOLO(args.model)

    #main detection loop
    while True: #update to take Signal() for ip camera input
        fishes = model.predict(args.video, conf=args.conf, agnostic_nms=True, stream=True)[0]
        for fish in fishes:
            box = fish.boxes
            prob= fish.probs
        frame = fishes.plot()
        detections = sv.Detections.from_yolov8(result)

        cv2.imshow('output', frame)

if __name__=='__main__':
    args()
    main()


###Useful Links
#https://docs.ultralytics.com/modes/predict/
#https://docs.ultralytics.com/modes/track/#tracking

