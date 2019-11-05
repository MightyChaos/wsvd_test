import cv2
import argparse
import os
import numpy as np

# from adapt.trainKernel import array2im

from networks.flownet2 import pred_flo, load_flownet
from yt3d.filterflow import resize_flow
from yt3d.utils import encodeFloatArray2Img, decodeBinaryArray


def saveDepthMap(file, arr):
    x, offset, scale = encodeFloatArray2Img(arr)
    np.savez_compressed(file, d=x, o=offset, s=scale)


def loadDepthMap(file):
    data = np.load(file)
    d = data['d'].astype(np.float32) * (data['s'] / 255) + data['o']
    return d


def array2im(input, amin=0, amax=1):
    # first normalize array
    # amin = input.min()
    # amax = input.max()
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
        from adapt.train_tempdepth2 import TempDepthKernel
        # pretrain_ckpt = '/home/chaoyang/exp/redweb_resnet/model_cur'
        # ckpt = '/home/chaoyang/exp/yt3d_tempdepth_2frame/model_cur.pth'
        self.test_kernel = TempDepthKernel(ckpt, imH, imW)
        # test_kernel.init_weights()
        self.test_kernel.load_model(ckpt)
        self.test_kernel.cuda()
        self.test_kernel.eval()
        self.imH = imH
        self.imW = imW

    def pred(self, im1, im2, flo):
        input_H = im1.shape[0]
        input_W = im1.shape[1]
        # im1 = cv2.resize(im1, (self.imW, self.imH))
        # im2 = cv2.resize(im2, (self.imW, self.imH))
        # flo = resize_flow(flo.transpose(1,2,0), self.imH, self.imW).transpose(2,0,1)
        pred_d = self.test_kernel.forward_pred(im1, im2, flo)
        # pred_d = cv2.resize(pred_d, (input_W, input_H))
        return pred_d


if __name__ == '__main__':
    '''
    usage:
        CUDA_VISIBLE_DEVICES=1 nice -10 python demo_video.py
    '''
    imH = 384
    imW = 704
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--vid_file',
        type=str,
        default='test.mp4',
        help='input video file')
    parser.add_argument(
        '-o',
        '--out_dir',
        type=str,
        default='demo_output',
        help='output folder')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        default='./ckpts/yt3d_tempdepth_2frame/model_cur',
        help='model location')

    args = parser.parse_args()
    # create output folder
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # load models:
    # 1. load flownet
    flow_net = load_flownet()
    flow_net.eval()
    flow_net.cuda()
    # 2. load depth net
    depth_net = DepthPred(ckpt=args.model_path)

    # iterate over input video
    cap = cv2.VideoCapture(args.vid_file)
    ret, frame_cur = cap.read()

    frame_cur = cv2.resize(frame_cur[..., ::-1], (imW, imH))
    cur_fid = 0
    while (cap.isOpened()):
        ret, frame_next = cap.read()
        if ret:
            frame_next = cv2.resize(frame_next[..., ::-1], (imW, imH))
            flo = pred_flo(flow_net, [frame_cur], [frame_next], 0)[0]
            pred_d = depth_net.pred(frame_cur, frame_next, flo)
            pred_q = 1 / pred_d  # convert to inverse depth
            d_im = array2im(pred_q, amin=0, amax=1)

            print('min: %f, max: %f' % (pred_q.min(), pred_q.max()))

            saveDepthMap(
                os.path.join(args.out_dir, '%07d.npz' % cur_fid), pred_q)

            cv2.imwrite(
                os.path.join(args.out_dir, '%07d.jpg' % cur_fid),
                np.concatenate([frame_cur, d_im], 1)[..., ::-1])

            frame_cur = frame_next
            cur_fid += 1
        else:
            break

    cap.release()
