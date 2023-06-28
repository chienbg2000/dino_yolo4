import numpy as np
from vidgear.gears import CamGear
import cv2

class VideoHandler:
    def __init__(self, youtube_url, net, output_layers, labels):
        self.stream = CamGear(source=youtube_url, stream_mode=True, backend="FFmpeg").start()
        self.net = net
        self.output_layers = output_layers
        self.labels = labels

    def read_frame(self):
        return self.stream.read()

    def crop_roi(self, frame, roi_x, roi_y, roi_width, roi_height):
        return frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    def detect_objects(self, roi , max_y):
        height, width, _ = roi.shape

        blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    if(y+h <= max_y):
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])



        return class_ids, confidences, boxes
    @staticmethod
    def show_frame(frame):
        cv2.imshow("YouTube Video", frame)
        cv2.waitKey(1)

    def stop(self):
        self.stream.stop()
