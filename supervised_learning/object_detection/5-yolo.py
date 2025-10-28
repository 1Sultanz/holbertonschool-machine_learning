#!/usr/bin/env python3
"""Class Yolo"""
import tensorflow.keras as K
import numpy as np
import cv2
import os


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
            grid_height, grid_width, anchor_boxes, _ = output.shape

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
            c_y = np.tile(c_y, [1, grid_width, anchor_boxes])

            anchors_w = self.anchors[i, :, 0]
            anchors_h = self.anchors[i, :, 1]

            anchors_w = anchors_w.reshape(1, 1, anchor_boxes)
            anchors_h = anchors_h.reshape(1, 1, anchor_boxes)

            t_x = 1 / (1 + np.exp(-t_x))
            t_y = 1 / (1 + np.exp(-t_y))

            b_x = t_x + c_x
            b_y = t_y + c_y

            b_x /= grid_width
            b_y /= grid_height

            b_w = anchors_w * np.exp(t_w)
            b_h = anchors_h * np.exp(t_h)

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

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter Boxes"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            box_classes_i = np.argmax(scores, axis=-1)
            box_scores_i = np.max(scores, axis=-1)

            filtering_mask = box_scores_i >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(box_classes_i[filtering_mask])
            box_scores.append(box_scores_i[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Non-max Suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        def iou(box, boxes):
            """IOU"""
            x1, y1, x2, y2 = box
            x1s, y1s, x2s, y2s = (boxes[:, 0], boxes[:, 1],
                                  boxes[:, 2], boxes[:, 3])
            inter_x1 = np.maximum(x1, x1s)
            inter_y1 = np.maximum(y1, y1s)
            inter_x2 = np.minimum(x2, x2s)
            inter_y2 = np.minimum(y2, y2s)
            inter_area = (np.maximum(0, inter_x2 - inter_x1) *
                          np.maximum(0, inter_y2 - inter_y1))
            box_area = (x2 - x1) * (y2 - y1)
            boxes_area = (x2s - x1s) * (y2s - y1s)
            union_area = box_area + boxes_area - inter_area
            iou = inter_area / union_area
            return iou

        unique_classes = np.unique(box_classes)
        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]
            sorted_idx = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sorted_idx]
            cls_scores = cls_scores[sorted_idx]
            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_scores.append(cls_scores[0])
                predicted_box_classes.append(cls)
                if len(cls_boxes) == 1:
                    break
                ious = iou(cls_boxes[0], cls_boxes[1:])
                mask = ious < self.nms_t
                cls_boxes = cls_boxes[1:][mask]
                cls_scores = cls_scores[1:][mask]
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)
        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load Images"""
        images = []
        image_paths = []
        files = os.listdir(folder_path)
        files.sort()
        for filename in files:
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                image_paths.append(image_path)
        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess Images"""
        pimages = []
        image_shapes = []
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            resized = cv2.resize(image, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
