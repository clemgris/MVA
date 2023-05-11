import torch, torchvision
import numpy as np
import os, json, cv2, random
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from torchvision import datasets

models =["COCO-Detection/retinanet_R_101_FPN_3x.yaml","COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml","COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml","COCO-Detection/retinanet_R_50_FPN_3x.yaml"]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

# return all the bounding boxed of bird found
def predictors(config_list):
    predictors = []
    for config in config_list:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
        predictor = DefaultPredictor(cfg)
        predictors.append(predictor)
    return predictors

# return the boudning box from a predictor
def detect_birds(predictor, im):
    outputs = predictor(im)
    if len(outputs["instances"].scores) != 0:
        probas = outputs["instances"].scores.cpu().numpy()
        labels = (outputs["instances"].pred_classes.cpu().numpy()==14)
        probas[~labels] = 0
        outputs["instances"].pred_boxes.tensor.cpu().numpy()
        idx = np.argmax(probas)
        proba = probas[idx]
        best_box = np.floor(outputs["instances"].pred_boxes.tensor.cpu().numpy()[idx]).astype('int')
        return proba, best_box
    else:
        return -1, [0,0,0,0]

# return the bounding box with the highest probability
def best_box(predictors, im):
    best_proba = -1
    best_bb = [0,0,0,0]
    for predictor in predictors:
        proba, bb = detect_birds(predictor,im)
        if proba > best_proba:
            best_proba = proba
            best_bb = bb
    return best_proba, best_bb

# return cropped image from the bounding box
def crop(im, bb):
    if sum(bb) ==0:
      return im
    left, top, right, bottom = bb
    cropped = im[top:bottom, left:right]
    return cropped

segmentations = ['COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml']
predictors_segmentation = predictors(segmentations )

# Create folders
path = "/content/drive/MyDrive/recvis22_a3/"
os.mkdir(path + "bird_dataset_centered/")
os.mkdir(path + "bird_dataset_centered/train_images")
os.mkdir(path + "bird_dataset_centered/val_images")
os.mkdir(path + "bird_dataset_centered/test_images/mistery_category")

# Center training set
for f in os.listdir(path + "bird_dataset/train_images/"):
    os.mkdir(path + "bird_dataset_centered/train_images/" + f)
    for im_path in os.listdir(path + "bird_dataset/train_images/" + f + "/"):
        im = cv2.imread(path + "bird_dataset/train_images/" +  f + "/" + im_path)
        _, bb = best_box(predictors_segmentation, im)
        im_cropped = crop(im, bb)
        cv2.imwrite(path + "bird_dataset_centered/train_images/" +  f + "/" + im_path, im_cropped)

# Center validation set
for f in os.listdir(path + "bird_dataset/val_images/"):
    os.mkdir(path + "bird_dataset_centered/val_images/" + f)
    for im_path in os.listdir(path + "bird_dataset/val_images/" + f + "/"):
        im = cv2.imread(path + "bird_dataset/val_images/" +  f + "/" + im_path)
        _, bb = best_box(predictors_segmentation, im)
        im_cropped= crop(im, bb)
        cv2.imwrite(path + "bird_dataset_centered/val_images/" +  f + "/" + im_path, im_cropped)

# Center testing set
for im_path in os.listdir(path + "bird_dataset/test_images/mistery_category/"):
    im = cv2.imread(path + "bird_dataset/test_images/mistery_category/" + im_path)
    _, bb = best_box(predictors_segmentation, im)
    im_cropped= crop(im, bb)
    cv2.imwrite(path + "bird_dataset_centered/test_images/mistery_category/" + im_path, im_cropped)
