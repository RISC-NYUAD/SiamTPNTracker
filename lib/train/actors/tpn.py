from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
import numpy as np

class SiamTPNActor(BaseActor):
    def __init__(self, net, objective, loss_weight, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.cfg = cfg
        self.grids = self._generate_anchors(cfg.MODEL.ANCHOR.NUM, cfg.MODEL.ANCHOR.FACTOR, cfg.MODEL.ANCHOR.BIAS)

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
        #print(grids)
        return grids

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss 
            status  -  dict containing detailed losses
        """
        # forward pass
        output = self.forward_pass(data) 

        target = {}
        target['anno'] = data['search_anno'][0]  # (batch, 4) (x1,y1,w,h)
        target['label'] = data['search_label'][0]  # B, 400 (100)
        target['centerness'] = data['centerness'][0]

        # compute losses
        loss, status = self.compute_losses(output, target)

        return loss, status

    def forward_pass(self, data):
        train_imgs = data['template_images']
        test_imgs = data['search_images']


        output = self.net(train_imgs[0], test_imgs[0])

        return output # B, N 4, B, N 1

    def compute_losses(self, output, target):

        pred_class, pred_boxes = output
        # Get class
        gt_class = target['label'].view(-1)
        cls_loss = self.objective['ce'](pred_class, gt_class)

        gt_centerness = target['centerness'].view(-1)
        mask = (gt_centerness > 0).to(pred_class.dtype)

        # Get boxes
        B, N, _ = pred_boxes.shape
        grids = self.grids[None,...].repeat(B, 1, 1).to(pred_boxes.device)
        lt = grids[:,:,:2] - pred_boxes[:,:,:2]
        rb = grids[:,:,:2] + pred_boxes[:,:,2:]
        pred_boxes = torch.cat([lt, rb], -1).view(-1, 4)

        gt_boxes = box_xywh_to_xyxy(target['anno']).clamp(min=0.0, max=1.0)
        gt_boxes = gt_boxes[:,None,:].repeat(1, N, 1).view(-1, 4)

        # compute giou and iou 
        giou_loss, iou = self.objective['giou'](pred_boxes, gt_boxes)   # (BN,4) (BN,4)
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes, gt_boxes, reduction='none')  # (BN,4) (BN,4)

        giou_loss = (giou_loss * mask).sum() / mask.sum()
        l1_loss = (l1_loss.sum(1) * mask).sum() / mask.sum()
        mean_iou = (iou.detach() * mask).sum() / mask.sum()
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss 

        loss = loss + self.loss_weight['ce'] * cls_loss
        status = {"Loss/total": loss.item(),
            "Loss/giou": giou_loss.item(),
            "Loss/l1": l1_loss.item(),
            "IoU": mean_iou.item()}
        status["Loss/cls"] = cls_loss.item()
        return loss, status

