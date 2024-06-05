from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("yolov8m-seg.pt")
