#!/usr/bin/env python3
"""Initialize Yolo"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """This class uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class Constructor"""
        self.model = K.models.load_model(model_path)
        self.class_names = []
        with open(classes_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.class_names.append(line)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Function to process outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = outputs.shape
            
            box_confidence = output[..., 4:5]
            box_class_prob = output[..., 5:]

            box_confidence = 1 / (1 + np.exp(-box_confidence))
            box_class_prob = 1 / (1 + np.exp(-box_class_prob))

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            c_x = np.arange(grid_width).reshape(1, grid_width, 1)
            c_x = np.tile(c_x, [grid_height, 1, anchor_boxes])

            c_y = np.arange(grid_height).reshape(grid_height, 1, 1)
            c_y = np.tile(grid_height, [1, grid_width, anchor_boxes])

            anchors_w = self.anchors[i, :, 0]
            anchors_h = self.anchors[i, :, 1]

            anchors_w = anchors_w.reshape(1, 1, anchor_boxes)
            anchors_h = anchors_h.reshape(1, 1, anchor_boxes)

            t_x = 1 / (1 + np.exp(-t_x))
            t_y = 1 / (1 + np.exp(-t_y))

            b_x = t_x + c_x
            b_y = t_y + c_y

            b_w = anchors_w * np.exp(t_w)
            b_h = anchors_h * np.exp(t_y)

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            b_w /= input_width
            b_h /= input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)
