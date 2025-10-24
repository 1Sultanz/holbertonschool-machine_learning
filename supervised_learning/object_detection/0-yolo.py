#!/usr/bin/env python3
"""Initialize Yolo"""
import tensorflow.keras as K


def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
    """This function uses the Yolo v3 algorithm to perform object detection"""
    self.model = K.models.load_model(model_path)
    self.classes_names = []
    with open(classes_path, 'r') as f:
        for line in f:
            line = line.strip()
            self.classes_names.append(line)
    self.class_t = class_t
    self.nms_t = nms_t
    self.anchors = anchors
