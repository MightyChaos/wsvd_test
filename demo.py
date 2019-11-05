import cv2
import sys
import os
import numpy as np

import torch

from networks.flownet2 import pred_flo, load_flownet
from networks.resdepth import TempDepthNet

def array2im(input):
    # first normalize array
    amin = input.min()
    amax = input.max()
    x = np.clip((input - amin) / (amax - amin + 1e-30) * 255, 0,
                255).astype(np.uint8)
    # covert to image according to colormap
    colormap = cv2.COLORMAP_JET
    im = cv2.applyColorMap(x, colormap)
    return im


class DepthPred():
    def __init__(self,
                 imH=384,
                 imW=704,
                 ckpt='ckpts/yt3d_tempdepth_2frame/model_cur'):
        
        self.depth_net = TempDepthNet(imH, imW)
        self.depth_net.load_model(ckpt)
        self.depth_net.cuda()
        self.depth_net.eval()

        self.flow_net = load_flownet()
        self.flow_net.eval()
        self.flow_net.cuda()

        self.imH = imH
        self.imW = imW

    def pred(self, im1_np, im2_np):
        # resize image to match the training set stat.
        im1_np = cv2.resize(im1_np, (self.imW, self.imH))
        im2_np = cv2.resize(im2_np, (self.imW, self.imH))
        # 1. predict flow with flownet2.0
        flo_np = pred_flo(self.flow_net, [im1_np], [im2_np], 0)[0]

        # 2. feed 2 images and the flow to depth estimator
        im1 = torch.from_numpy(np.transpose(
            im1_np, [2, 0, 1]).astype(np.float32))
        im2 = torch.from_numpy(np.transpose(
            im2_np, [2, 0, 1]).astype(np.float32))
        flo = torch.from_numpy(flo_np)
        with torch.no_grad():
            im1 = im1.unsqueeze(0).cuda()/255
            im2 = im2.unsqueeze(0).cuda()/255
            flos = flo.unsqueeze(0).cuda()
            log_depth_pred = self.depth_net.forward(im1, im2, flos)
            pred_d = log_depth_pred.exp()

        pred_d = pred_d.data.cpu().numpy()[0, ...]
        return pred_d


if __name__ == '__main__':
    '''
    usage:
        CUDA_VISIBLE_DEVICES=0 nice -10 python demo.py demo/im1.png demo/im2.png
    '''
    im_files = sys.argv[1:3]
    ims = [cv2.imread(im_file)[..., ::-1] for im_file in im_files]

    # load models:
    depth_estimator = DepthPred()

    pred_d = depth_estimator.pred(ims[0], ims[1])
    pred_q = 1 / pred_d  # convert to inverse depth
    d_im = array2im(pred_q)


    cv2.imwrite(
        'demo_output.png',
        np.concatenate([ims[0], d_im], 1)[..., ::-1])
