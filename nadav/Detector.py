from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import numpy as np


class Detector:
    def __init__(self,model_type = "OD"):
        self.cfg = get_cfg()
        self.model_type = model_type
        if model_type == "OD":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":#instance seg
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP":#keypoints detections
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "LVIS":#LVIS detections
            self.cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == "PS":#keypoints detections
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticsSegmentation/panoptic_fpn_R_101_3xyaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticsSegmentation/panoptic_fpn_R_101_3xyaml")
        # Lower the threshold to detect more objects
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust threshold if necessary
        self.cfg.MODEL.DEVICE = "cuda"  # Use GPU if available, otherwise set to "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        # Read the image
        image = cv2.imread(imagePath)
        if self.model_type != "PS":
            # Make predictions
            predictions = self.predictor(image)

            # Create a Visualizer object
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            vis = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=metadata,
                instance_mode=ColorMode.IMAGE  # Keep the original colors
            )

            # Draw predictions on the image
            output = vis.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            predictions, segmentInfo = self.predictor(image)["panptic_seg"]
            vis = Visualizer(image[:,:,::-1],MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = vis.draw_panoptic_seg_predictions(predictions.to("cpu"),segmentInfo)
        # Display the result
        cv2.imshow("Results", output.get_image()[:, :, ::-1])  # Convert RGB back to BGR for OpenCV
        cv2.waitKey(0)
    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        if (cap.isOpened()==False):
            print ("Error openong the file..")
            return
        (sucess,image) = cap.read()

        while sucess:

            if self.model_type != "PS":
                # Make predictions
                predictions = self.predictor(image)

                # Create a Visualizer object
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                vis = Visualizer(
                    image[:, :, ::-1],  # Convert BGR to RGB
                    metadata=metadata,
                    instance_mode=ColorMode.IMAGE  # Keep the original colors
                )

                # Draw predictions on the image
                output = vis.draw_instance_predictions(predictions["instances"].to("cpu"))
            else:
                predictions, segmentInfo = self.predictor(image)["panptic_seg"]
                vis = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = vis.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            cv2.imshow("Result",output.get_image()[:,:,::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (sucess,image) = cap.read()
