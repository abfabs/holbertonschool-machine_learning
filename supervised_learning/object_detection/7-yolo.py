#!/usr/bin/env python3
"""7-yolo.py"""

import os
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    """YOLO v3 object detection class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path, compile=False)

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

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_x = np.tile(c_x, (grid_h, 1, anchor_boxes))

            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            c_y = np.tile(c_y, (1, grid_w, anchor_boxes))

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape((1, 1, anchor_boxes))
            anchor_h = self.anchors[i, :, 1].reshape((1, 1, anchor_boxes))

            b_w = (np.exp(t_w) * anchor_w) / input_w
            b_h = (np.exp(t_h) * anchor_h) / input_h

            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes using objectness score and class probability"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered boxes"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[cls_indices]
            cls_box_classes = box_classes[cls_indices]
            cls_box_scores = box_scores[cls_indices]

            ranked = np.argsort(cls_box_scores)[::-1]
            cls_boxes = cls_boxes[ranked]
            cls_box_classes = cls_box_classes[ranked]
            cls_box_scores = cls_box_scores[ranked]

            while len(cls_box_scores) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls_box_classes[0])
                predicted_box_scores.append(cls_box_scores[0])

                if len(cls_box_scores) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                box_area = (
                    (cls_boxes[0, 2] - cls_boxes[0, 0]) *
                    (cls_boxes[0, 3] - cls_boxes[0, 1])
                )
                other_areas = (
                    (cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                    (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                )

                union = box_area + other_areas - intersection
                iou = intersection / union

                keep = np.where(iou < self.nms_t)[0]

                cls_boxes = cls_boxes[keep + 1]
                cls_box_classes = cls_box_classes[keep + 1]
                cls_box_scores = cls_box_scores[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder"""
        images = []
        image_paths = []

        for filename in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path):
                image = cv2.imread(path)
                if image is not None:
                    images.append(image)
                    image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and rescale images for YOLO input"""
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            pimages.append(resized / 255.0)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with boundary boxes, class names, and box scores"""
        image_copy = image.copy()

        for i, box in enumerate(boxes):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cv2.rectangle(
                image_copy,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            text = "{} {:.2f}".format(class_name, score)

            cv2.putText(
                image_copy,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image_copy)
        key = cv2.waitKey(0)

        if key == ord("s"):
            if not os.path.isdir("detections"):
                os.makedirs("detections")

            save_path = os.path.join("detections", file_name)
            cv2.imwrite(save_path, image_copy)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Predict objects in all images inside a folder"""
        predictions = []
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages, verbose=0)

        for i, image in enumerate(images):
            image_outputs = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs,
                image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes,
                box_confidences,
                box_class_probs
            )

            box_predictions, predicted_box_classes, predicted_box_scores = (
                self.non_max_suppression(
                    filtered_boxes,
                    box_classes,
                    box_scores
                )
            )

            predictions.append(
                (
                    box_predictions,
                    predicted_box_classes,
                    predicted_box_scores
                )
            )

            self.show_boxes(
                image,
                box_predictions,
                predicted_box_classes,
                predicted_box_scores,
                os.path.basename(image_paths[i])
            )

        return predictions, image_paths
