import argparse
import sys
sys.path.append('./')
import torch
from lib.models.siamtpn.track import build_network
from lib.config.default import cfg, update_config_from_file
from lib.models.siamtpn.track import build_network
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from lib.test.evaluation.environment import env_settings


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--config', type=str, default='shufflenet_l345_192', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz, requires_grad=False)
    kernel = torch.randn(bs, 192, 5, 5, requires_grad=False)
    return img_patch, kernel


class Tracker_z(nn.Module):
    def __init__(self, backbone, fpn, head):
        super(Tracker_z, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head

    def forward(self, img: torch.Tensor, kernel: torch.Tensor):
        feat = self.backbone(img)  # BxCxHxW
        feat = self.fpn(feat)
        pred_cls, pred_box = self.head(feat, kernel)
        return pred_cls, pred_box


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    save_name = "./results/onnx/backbone_fpn_head_x.onnx"

    """update cfg"""
    args = parse_args()
    yaml_fname = 'experiments/{}.yaml'.format(args.config)
    update_config_from_file(cfg, yaml_fname)

    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.SEARCH_SIZE
    # build the stark model
    model = build_network(cfg)
    # load checkpoint
    checkpoint_name = os.path.join("results/checkpoints/train/shufflenet_l345_192/TransPatchTrack_ep0100.pth.tar")
    model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=False)
    model.eval()
    """ rebuild the inference-time model """
    backbone = model.backbone
    fpn = model.fpn
    head = model.head
    torch_model = Tracker_z(backbone, fpn, head).eval()
    #print(torch_model)
    # get the input
    img_x, kernel = get_data(bs, z_sz)
    # forward the input
    torch_outs = torch_model(img_x, kernel)
    torch.onnx.export(torch_model,  # model being run
                      (img_x,kernel),  # model input (or a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['img_x','kernel'],  # the model's input names
                      output_names=['pred_cls','pred_box'],  # the model's output names
                      )
    # latency comparison
    N = 100
    """########## inference with the pytorch model ##########"""
    s = time.time()
    for i in range(N):
        _ = torch_model(img_x, kernel)
    e = time.time()
    print("pytorch model average latency: %.2f ms" % ((e - s) / N * 1000))
    """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_name, providers=['OpenVINOExecutionProvider'])

    # compute ONNX Runtime output prediction
    ort_inputs = {'img_x': to_numpy(img_x), 'kernel': to_numpy(kernel)}
    # print(onnxruntime.get_device())
    # warmup
    for i in range(10):
        ort_outs = ort_session.run(None, ort_inputs)
    s = time.time()
    for i in range(N):
        ort_outs = ort_session.run(None, ort_inputs)
    e = time.time()
    print("onnx model average latency: %.2f ms" % ((e - s) / N * 1000))
    # compare ONNX Runtime and PyTorch results
    for i in range(2):
        np.testing.assert_allclose(to_numpy(torch_outs[i]), ort_outs[i], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
