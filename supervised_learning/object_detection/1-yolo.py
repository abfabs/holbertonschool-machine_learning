#!/usr/bin/env python3
"""1-yolo.py"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """YOLO v3 object detection class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the Darknet model

        Args:
            outputs: list of numpy.ndarrays containing predictions for one image
            image_size: numpy.ndarray containing original image size
                        [image_height, image_width]

        Returns:
            boxes, box_confidences, box_class_probs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h, input_w = K.backend.int_shape(self.model.input)[1:3]
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Grid offsets
            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, (grid_h, 1, anchor_boxes))

            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, (1, grid_w, anchor_boxes))

            # Sigmoid helper
            sigmoid = lambda x: 1 / (1 + np.exp(-x))

            # Box center positions
            b_x = (sigmoid(t_x) + c_x) / grid_w
            b_y = (sigmoid(t_y) + c_y) / grid_h

            # Box width and height
            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            b_w = (np.exp(t_w) * anchor_w) / input_w
            b_h = (np.exp(t_h) * anchor_h) / input_h

            # Convert to corner coordinates relative to original image size
            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            # Box confidence
            box_confidence = sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Class probabilities
            box_class_prob = sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
