import cv2
from ultralytics import YOLO
from torch import cuda
import numpy as np

class objectDetection:
    def __init__(self, confidence_activation, weights_file, names_file):
        self.confidence = confidence_activation
        self.nmsThreshold = 0.3

        self.weights_file = weights_file

        self.class_names = []
        with open(names_file, "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

    def loadNN(self):
        self.model = YOLO(self.weights_file, task='detect')
        if cuda.is_available():
            self.proc = '0'
        else:
            self.proc = 'cpu'

    def detection(self, frame):
        results = self.model.predict(frame, conf = self.confidence, iou=self.nmsThreshold, 
                                     imgsz=416, device = self.proc, verbose = False)

        classes = np.array(results[0].boxes.cls.tolist())
        scores = np.array(results[0].boxes.conf.tolist())
        boxes = np.array(results[0].boxes.xyxy.tolist())
        
        return classes, scores, boxes

class outputTesting:
    def __init__(self, names_file):
        self.color = (0, 255, 255)
        self.class_names = []
        with open(names_file, 'r') as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

    def testOutputFrames(self, frame, tracked_fish): 
        for t in tracked_fish:
            label = "%s" % (self.class_names[t[1].astype(int)] + ", id: " + str(t[0]) + ", max_score: " + str(t[2]))
            cv2.rectangle(frame, (t[3][0].astype(int), t[3][1].astype(int)), 
                          (t[3][2].astype(int), t[3][3].astype(int)), self.color, 2)
            cv2.putText(frame, label, (t[3][0].astype(int), t[3][1].astype(int) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, self.color, 6)
        #frame = cv2.resize(frame, (360,640)) #good for lotic.demo vid since its 4k
        #cv2.imshow("detections", frame) #For visualizing live detections, uncomment this line
