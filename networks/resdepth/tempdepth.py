from .depthnet import resnet50, ImEncoder, DepthDecoder, PyramidFeaWarp, FloImsEarlyFusion
import torch.nn as nn
import torch
import itertools
import numpy as np

class TempDepthNet(nn.Module):
    def __init__(self, imH=384, imW=704):
        super(TempDepthNet, self).__init__()
        self.resnet = resnet50(False)
        self.resnet.load_model('ckpts/resnet50-19c8e357.pth')
        self.resnet.eval()

        self.im_encoder = ImEncoder()
        self.depth_decoder = DepthDecoder()

        self.flo_fuse_encoder = FloImsEarlyFusion()
        
        self.pyrFeaWarp = PyramidFeaWarp(imH, imW)

    def forward(self, im1, im2, flo):
        b = im1.size(0)
        with torch.no_grad():
            fea1, fea2, fea3, fea4 = self.resnet.extract_feat(
                torch.cat((im1, im2), 0))
        im_fea_list = self.im_encoder([fea4, fea3, fea2, fea1])
        im1_fea_list = [f[:b, ...] for f in im_fea_list]
        im2_fea_list = [f[b:, ...] for f in im_fea_list]
        im2_warp_fea_list = self.pyrFeaWarp(im2_fea_list[::-1], flo)[::-1]
        fea_list = self.flo_fuse_encoder(flo, im1_fea_list, im2_warp_fea_list)
        pred = self.depth_decoder(fea_list)
        return pred.squeeze(1)

    def parameters(self):
        return itertools.chain(
            self.depth_decoder.parameters(),
            self.im_encoder.parameters(),
            self.flo_fuse_encoder.parameters())

    def load_model(self, file_path):
        file_path = file_path.split('.pth')[0]
        self.depth_decoder.load_state_dict(
            torch.load(file_path+'_depth_decoder.pth'))
        self.im_encoder.load_state_dict(
            torch.load(file_path+'_im_encoder.pth'))
        self.flo_fuse_encoder.load_state_dict(
            torch.load(file_path+'_flo_encoder.pth'))

    def save_model(self, file_path):
        self.cpu()
        file_path = file_path.split('.pth')[0]
        torch.save(self.depth_decoder.state_dict(),
                    file_path+'_depth_decoder.pth')
        torch.save(self.im_encoder.state_dict(),
                    file_path+'_im_encoder.pth')
        torch.save(self.flo_fuse_encoder.state_dict(),
                    file_path+'_flo_encoder.pth')

        self.cuda()


if __name__ == '__main__':
    # net = DepthNet()
    # pred = net.forward(torch.randn(1,3,384,384))
    # print(pred.size())
    net = TempDepthNet(
        '/home/chaoyang/exp/yt3d_resnet_redweb_gradloss/model_cur',
        384, 384)
    net.cuda()
    pred = net.forward(torch.randn(4, 3, 384, 384).cuda(),
                       torch.randn(4, 3, 384, 384).cuda(),
                       torch.randn(4, 2, 384, 384).cuda())
    print(pred.size())
