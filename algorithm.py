import os
import torch
import cv2
import numpy as np
from pathlib import Path

from models.common import DetectMultiBackend, Classifier
from models.scrfd import SCRFD
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


class Algorithm:
    def __init__(self):
        # init cloth detector config
        self.device = select_device('cpu')
        self.config_neiku = {'weights':['checkpoints/neiku_epoch300_s.pt', 'checkpoints/neiku_epoch300_m.pt'], 'two_stage':False, 'cls_weights':[]}
        self.config_xiongzhao = {'weights':['checkpoints/xiongzhao_epoch300_m.pt', 'checkpoints/xiongzhao_epoch300_s.pt'], 'two_stage':False, 'cls_weights':[]}

        # Initialize/load cloth detector model
        weights = [os.path.join(os.path.dirname(__file__), weight) for weight in  self.config_neiku['weights']]
        print(weights)
        self.model_neiku = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
        
        weights = [os.path.join(os.path.dirname(__file__), weight) for weight in  self.config_xiongzhao['weights']]
        self.model_xiongzhao = DetectMultiBackend(weights, device=self.device, dnn=False, data=None, fp16=False)
        
        # Second-stage classifier (optional)
        if self.config_neiku['two_stage']:
            cls_weights = os.path.join(os.path.dirname(__file__),self.config_neiku['cls_weights'])
            self.ClassifyNet_neiku = Classifier(self.config_neiku, cls_weights)
            
        if self.config_xiongzhao['two_stage']:
            cls_weights = os.path.join(os.path.dirname(__file__),self.config_xiongzhao['cls_weights'])
            self.ClassifyNet_xiongzhao = Classifier(self.config_xiongzhao, cls_weights)
        
        # init face detector
        self.detector = SCRFD(model_file=os.path.join(os.path.dirname(__file__), 'checkpoints/model.onnx'))
        self.detector.prepare(-1)
            
    def process(self, im, option): # set thresholds in options
        method = option.get("method")
        threshold_face = option.get("thres_face")
        threshold_cloth = option.get("thres_cloth")
        
        method = 'all' if method is None else method
        threshold_face = 0.48 if threshold_face is None else threshold_face
        threshold_cloth = 0.8 if threshold_cloth is None else threshold_cloth
        img_size = 1024
        im = im[0]
        im = cv2.resize(im,(img_size,img_size))
        
        with torch.no_grad():
            res = {}
            cloth_result = []
            face_result = []
            # detect face
            if method == "all" or method == "face":
                bboxes, _ = self.detector.detect(im[:,:,::-1], threshold_face, input_size = (img_size, img_size))
                for bbox in bboxes:
                    bbox_dict = {'x1': max(int(bbox[0]), 0), 'y1': max(int(bbox[1]), 0), 'x2': int(bbox[2]), 'y2': int(bbox[3])} # x1 and y1 may < 0
                    face_result.append(bbox_dict)
                res['face'] = 1 if len(face_result) else 0
            
            if method == "all" or method == "cloth":
                # Pre process
                im = im.transpose((2, 0, 1))  # HWC to CHW
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model_neiku.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                
                # Neiku inference 
                pred_neiku = self.model_neiku(im, augment=True, visualize=False)
                pred_neiku = non_max_suppression(pred_neiku, 0.001, iou_thres=0.6, classes=None, agnostic=False, max_det=1000)
                scale_coords(im.shape[2:], pred_neiku[0][:, :4], (img_size, img_size, 3))
                
                # xiongzhao inference
                pred_xiongzhao = self.model_xiongzhao(im, augment=True, visualize=False)
                pred_xiongzhao = non_max_suppression(pred_xiongzhao, 0.001, iou_thres=0.6, classes=None, agnostic=False, max_det=1000)
                scale_coords(im.shape[2:], pred_xiongzhao[0][:, :4], (img_size, img_size, 3))
                
                # Second-stage classifier (optional)
                if self.config_neiku['two_stage']:
                    pred_neiku = self.ClassifyNet_neiku(pred_neiku, im)
                if self.config_xiongzhao['two_stage']:
                    pred_xiongzhao = self.ClassifyNet_xiongzhao(pred_xiongzhao, im)
                
                # return results
                cloth_result = [item.cpu().numpy()[:4] for item in pred_neiku[0] if item[4] >= threshold_cloth] + [item.cpu().numpy()[:4] for item in pred_xiongzhao[0] if item[4] >= threshold_cloth]
                res['cloth'] = 1 if(cloth_result) else 0
                
            return res

