from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, transform_image_to_crop, sample_target_fast
# for debug
import numpy as np
import torch.nn.functional as F
import cv2
import os
import onnxruntime
from lib.test.tracker.utils import Preprocessor,PreprocessorX_onnx
from lib.utils.box_ops import clip_box


class SiamTPN(BaseTracker):
    def __init__(self, params):
        super(SiamTPN, self).__init__(params)
        self.cfg = params.cfg
        providers=['OpenVINOExecutionProvider']
        device = 'CPU_FP32'
        self.ort_sess_z = onnxruntime.InferenceSession("./results/onnx/backbone_fpn_z.onnx", providers=providers,
                                                       provider_options=[{'device_type' : device}])
        self.ort_sess_x = onnxruntime.InferenceSession("./results/onnx/backbone_fpn_head_x.onnx", providers=providers,
                                                       provider_options=[{'device_type' : device}])
        self.preprocessor = PreprocessorX_onnx()
        # for debug
        self.debug = self.params.debug
        self.frame_id = 0
        self.grids = self._generate_anchors(self.cfg.MODEL.ANCHOR.NUM, self.cfg.MODEL.ANCHOR.FACTOR, self.cfg.MODEL.ANCHOR.BIAS)
        self.window = self._hanning_window(self.cfg.MODEL.ANCHOR.NUM)
        self.hanning_factor = self.cfg.TEST.HANNING_FACTOR
        self.feat_sz_tar = self.cfg.MODEL.ANCHOR.NUM

    def initialize(self, image, info: dict):
        gt_box = info['init_bbox']
        z_patch_arr, _ = sample_target_fast(image, gt_box,self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr)
        ort_inputs = {'img_z': template}
        self.kernel = self.ort_sess_z.run(None, ort_inputs)[0]       
        self.state = info['init_bbox']
        self.frame_id = 0


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target_fast(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        search = self.preprocessor.process(x_patch_arr)
        ort_inputs = {'img_x': search,
                      'kernel': self.kernel
                      }
        raw_scores, boxes = self.ort_sess_x.run(None, ort_inputs)
        pred_boxes = boxes.reshape(-1, 4)
        lt = self.grids[:,:2] - pred_boxes[:,:2]
        rb = self.grids[:,:2] + pred_boxes[:,2:]
        lt = np.asarray(lt)
        rb = np.asarray(rb)
        pred_boxes = np.concatenate([lt, rb], -1)
        raw_scores = raw_scores[:,1]
        raw_scores = raw_scores * (1-self.hanning_factor) + self.hanning_factor * self.window
        ind = np.argmax(raw_scores)
        pred_box = pred_boxes[ind, :]
        pred_box = (pred_box * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=5)
            cv2.imshow("demo", image_BGR)
            key = cv2.waitKey(1)
            if key == ord('p'):
                cv2.waitKey(-1)

        # get the final box result
        H, W, _ = image.shape
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        #print(self.state)
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        x1,y1,x2,y2 = pred_box
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        #print(cx_real, cy_real, cx_prev, cy_prev, cx, cy)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _generate_anchors(self, num=20, factor=1, bias=0.5):
        """
        generate anchors for each sampled point
        """
        x = np.arange(num)
        y = np.arange(num)
        xx, yy = np.meshgrid(x, y) 
        xx = (factor * xx + bias) / num 
        yy = (factor * yy + bias) / num
        xx = torch.from_numpy(xx).view(-1).float()
        yy = torch.from_numpy(yy).view(-1).float()
        grids = torch.stack([xx, yy],-1) # N 2
        return np.array(grids.numpy())

    def _hanning_window(self, num):
        hanning = np.hanning(num)
        window = np.outer(hanning, hanning)
        return window.reshape(-1)

def get_tracker_class():
    return SiamTPN
