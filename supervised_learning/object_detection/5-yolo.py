#!/usr/bin/env python3
"""5-yolo.py"""

import os
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    """YOLO v3 object detection class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x, c_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            b_w = (anchor_w * np.exp(t_w)) / self.model.input.shape[1]
            b_h = (anchor_h * np.exp(t_h)) / self.model.input.shape[2]

            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.zeros((grid_h, grid_w, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes using objectness score and class probability"""
        filtered_boxes = np.empty((0, 4))
        box_classes = np.empty((0,), dtype=int)
        box_scores = np.empty((0,))

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes = np.concatenate(
                (filtered_boxes, boxes[i][mask]),
                axis=0
            )
            box_classes = np.concatenate(
                (box_classes, classes[mask]),
                axis=0
            )
            box_scores = np.concatenate(
                (box_scores, class_scores[mask]),
                axis=0
            )

        return filtered_boxes, box_classes, box_scores

    def iou(self, box1, box2):
        """Calculate intersection over union for two boxes"""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2

        x1 = np.maximum(b1_x1, b2_x1)
        y1 = np.maximum(b1_y1, b2_y1)
        x2 = np.minimum(b1_x2, b2_x2)
        y2 = np.minimum(b1_y2, b2_y2)

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - intersection

        return intersection / union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered boxes"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            class_indices = np.where(box_classes == cls)[0]
            class_boxes = filtered_boxes[class_indices]
            class_scores = box_scores[class_indices]

            while len(class_boxes) > 0:
                max_score_index = np.argmax(class_scores)

                box_predictions.append(class_boxes[max_score_index])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(class_scores[max_score_index])

                ious = np.array([
                    self.iou(class_boxes[max_score_index], box)
                    for box in class_boxes
                ])

                remove = np.where(ious > self.nms_t)[0]

                class_boxes = np.delete(class_boxes, remove, axis=0)
                class_scores = np.delete(class_scores, remove)

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder"""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(folder_path, filename)
                image = cv2.imread(path)

                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and rescale images for YOLO input"""
        pimages = []
        image_shapes = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_h, input_w),
                interpolation=cv2.INTER_CUBIC
            )

            pimages.append(resized / 255.0)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
