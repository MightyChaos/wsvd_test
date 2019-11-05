from .resnet import resnet50
import torch.nn as nn
import torch
import itertools
import numpy as np
from .warp import FloWarp

def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def convBlock(in_planes, out_planes, kernel_size, stride=1, batch_norm=True):
    model_list = [nn.Conv2d(in_planes, out_planes,
                    kernel_size, stride=stride, bias = not batch_norm, padding=kernel_size//2)]
    if batch_norm:
        model_list.append(nn.BatchNorm2d(out_planes))
    model_list.append(nn.ReLU(inplace=True))
    return nn.Sequential(*model_list)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
            conv3x3(inplanes, planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes)
            )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

def define_fuse_block():
    return nn.Sequential(
            ResBlock(256, 256),
            nn.UpsamplingBilinear2d(scale_factor=2))

def define_im_fea_proj_block(in_plane, out_plane=256):
    return nn.Sequential(
        conv3x3(in_plane, out_plane),
        ResBlock(out_plane, out_plane)
    )

class ImEncoder(nn.Module):
    def __init__(self):
        super(ImEncoder, self).__init__()
        self.model_list = nn.ModuleList([
                define_im_fea_proj_block(2048, 256),
                define_im_fea_proj_block(1024, 256),
                define_im_fea_proj_block(512, 256),
                define_im_fea_proj_block(256, 256)
            ])
    
    def forward(self, fea_list):
        out_fea_list = []
        for i in range(len(fea_list)):
            out_fea_list.append(self.model_list[i](fea_list[i]))
        return out_fea_list

class FloFusionEncoder(nn.Module):
    def __init__(self):
        super(FloFusionEncoder, self).__init__()
        self.conv_layers = nn.ModuleList([
            convBlock(2, 32, 7, 2, True),
            convBlock(32, 64, 5, 2, True),
            nn.Sequential(
                convBlock(64, 128, 3, 1, True),
                convBlock(128, 128, 3, 1, True) ),
            nn.Sequential(
                convBlock(128, 256, 3, 2, True),
                convBlock(256, 256, 3, 1, True) ),
            nn.Sequential(
                convBlock(256, 512, 3, 2, True),
                convBlock(512, 512, 3, 1, True) ),
            nn.Sequential(
                convBlock(512, 1024, 3, 2, True),
                convBlock(1024, 1024, 3, 1, True) ),
        ])
        
        self.fea_proj_layers = nn.ModuleList([
            define_im_fea_proj_block(1024, 128),
            define_im_fea_proj_block(512, 128),
            define_im_fea_proj_block(256, 128),
            define_im_fea_proj_block(128, 128)            
        ])

        self.fuser_layers = nn.ModuleList([
            define_im_fea_proj_block(256+128, 256),
            define_im_fea_proj_block(256+128, 256),
            define_im_fea_proj_block(256+128, 256),
            define_im_fea_proj_block(256+128, 256)
        ])

    def forward(self, flo, im_fea_list):
        # compute conv feats
        x = self.conv_layers[0](flo)
        x = self.conv_layers[1](x)
        fea1 = self.conv_layers[2](x)
        fea2 = self.conv_layers[3](fea1)
        fea3 = self.conv_layers[4](fea2)
        fea4 = self.conv_layers[5](fea3)
        fea_list = [fea4, fea3, fea2, fea1]
        # project to 128 channel
        flo_fea_list = []
        for i in range(4):
            flo_fea_list.append(
                self.fea_proj_layers[i](fea_list[i])
            )
        # fuse with im feats
        out_fea_list = []
        for i in range(4):
            # print(flo_fea_list[i].size())
            x = torch.cat((im_fea_list[i], flo_fea_list[i]), 1)
            out_fea_list.append(
                self.fuser_layers[i](x)
            )   
        return out_fea_list


def meshgrid(imH, imW):
    x = torch.from_numpy(np.arange(imW)).float()
    y = torch.from_numpy(np.arange(imH)).float()
    x = x.unsqueeze(0).repeat(imH, 1)
    y = y.unsqueeze(1).repeat(1, imW)
    return x, y

class FloEarlyFusionEncoder(nn.Module):
    def __init__(self, add_uv=False, imH=384, imW=704):
        super(FloEarlyFusionEncoder, self).__init__()
        self.add_uv = add_uv
        self.imH = imH
        self.imW = imW
        if self.add_uv:
            in_chan = 4
            u, v = meshgrid(imH, imW)
            uv = torch.stack((u/imW-.5, v/imH-.5), 0)
            self.register_buffer('uv', uv)
        else:
            in_chan = 2

        self.conv_layers = nn.ModuleList([
            convBlock(in_chan, 32, 7, 2, True),
            convBlock(32, 64, 5, 2, True),
            nn.Sequential(
                convBlock(64, 128, 3, 1, True),
                convBlock(128, 128, 3, 1, True) ),
            nn.Sequential(
                convBlock(128, 256, 3, 2, True),
                convBlock(256, 256, 3, 1, True) ),
            nn.Sequential(
                convBlock(256, 512, 3, 2, True),
                convBlock(512, 512, 3, 1, True) ),
            nn.Sequential(
                convBlock(512, 1024, 3, 2, True),
                convBlock(1024, 1024, 3, 1, True) ),
        ])
        
        self.flo_fea_proj_layers = nn.ModuleList([
            define_im_fea_proj_block(1024, 128),
            define_im_fea_proj_block(512, 128),
            define_im_fea_proj_block(256, 128),
            define_im_fea_proj_block(128, 128)            
        ])

        self.early_fuse_layers = nn.ModuleList([
            nn.Sequential(
                convBlock(256+128, 256, 3, 1, True),
                convBlock(256, 256, 3, 1, True)),
            nn.Sequential(
                convBlock(256+128+256, 512, 3, 1, True),
                convBlock(512, 512, 3, 1, True)),
            nn.Sequential(
                convBlock(256+128+512, 512, 3, 1, True),
                convBlock(512, 512, 3, 1, True)),
            nn.Sequential(
                convBlock(256+128+512, 1024, 3, 1, True),
                convBlock(1024, 1024, 3, 1, True)),
        ])

        self.downsample_layers = nn.ModuleList([
            convBlock(256, 256, 3, 2, True),
            convBlock(512, 512, 3, 2, True),
            convBlock(512, 512, 3, 2, True)
        ])

        self.fea_proj_layers = nn.ModuleList([
            define_im_fea_proj_block(1024, 256),
            define_im_fea_proj_block(512, 256),
            define_im_fea_proj_block(512, 256),
            define_im_fea_proj_block(256, 256)
        ])

    def forward(self, flo, im_fea_list):
        # compute conv feats
        bsize = flo.size(0)
        if self.add_uv:
            uv = self.uv.unsqueeze(0).expand(bsize, 2, self.imH, self.imW)
            x = torch.cat((flo, uv), 1)
            x = self.conv_layers[0](x)
        else:
            x = self.conv_layers[0](flo)
        x = self.conv_layers[1](x)
        fea1 = self.conv_layers[2](x)
        fea2 = self.conv_layers[3](fea1)
        fea3 = self.conv_layers[4](fea2)
        fea4 = self.conv_layers[5](fea3)
        fea_list = [fea4, fea3, fea2, fea1]
        # project to 128 channel
        flo_fea_list = []
        for i in range(4):
            flo_fea_list.append(
                self.flo_fea_proj_layers[i](fea_list[i])
            )
        # fuse with im feats
        fused_fea_list = [self.early_fuse_layers[0](
            torch.cat( (im_fea_list[3], flo_fea_list[3]), 1) )]
        for i in range(3):
            x = self.downsample_layers[i](fused_fea_list[i])
            x = torch.cat((x,
                           im_fea_list[2-i], 
                           flo_fea_list[2-i]), 1)
            fused_fea_list.append(
                self.early_fuse_layers[i+1](x)
            )
        out_fea_list = []
        for i in range(4):
            out_fea_list.append(
                self.fea_proj_layers[i](fused_fea_list[3-i])
            )   
        return out_fea_list


class FloImsEarlyFusion(nn.Module):
    def __init__(self):
        super(FloImsEarlyFusion, self).__init__()
        self.fuse_net = FloEarlyFusionEncoder()
        self.model_list = nn.ModuleList([
            define_im_fea_proj_block(512, 256),
            define_im_fea_proj_block(512, 256),
            define_im_fea_proj_block(512, 256),
            define_im_fea_proj_block(512, 256)
        ])
    def forward(self, flo,  im_fea_list, im_fea_warp_list):
        fea_list = []
        for i in range(4):
            fea_list.append(
                self.model_list[i](
                    torch.cat((im_fea_list[i], im_fea_warp_list[i]), 1)
                    ) 
                    )
        
        out_fea_list = self.fuse_net(flo, fea_list)
        return out_fea_list



class DepthDecoder(nn.Module):
    def __init__(self):
        super(DepthDecoder, self).__init__()
        self.fuser_blocks = nn.ModuleList([
            define_fuse_block(),
            define_fuse_block(),
            define_fuse_block(),
            define_fuse_block()
        ])

        self.output_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            conv3x3(256, 128),
            nn.ReLU(inplace=True),
            conv3x3(128, 1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, im_fea_list):
        x = self.fuser_blocks[0](im_fea_list[0])
        for i in range(1,len(im_fea_list)):
            x = self.fuser_blocks[i](x+im_fea_list[i])
        return self.output_layer(x)


class PyramidFeaWarp(nn.Module):
    def __init__(self, imH, imW):
        super(PyramidFeaWarp, self).__init__()
        self.levels = [2, 3, 4, 5]
        self.downsample = nn.AvgPool2d(2, stride=2)
        self.warp_layers = nn.ModuleList([
            FloWarp(imH//(2**k), imW//(2**k)) for k in self.levels
        ])
    
    def forward(self, fea_list, flo):
        # first downsample flo
        with torch.no_grad():
            flo_pyr = [flo]
            for i in range(5):
                flo_pyr.append(self.downsample(flo_pyr[-1])/2)
            flo_pyr = flo_pyr[2:6]
        # warp features
        fea_warp_list = []
        for i in range(4):
            fea_warp_list.append(
                self.warp_layers[i](fea_list[i], flo_pyr[i][:,0,:,:], flo_pyr[i][:,1,:,:])
            )
        return fea_warp_list



class DepthDiffDecoder(nn.Module):
    def __init__(self):
        super(DepthDiffDecoder, self).__init__()
    def forward(self, x):
        pass


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.resnet = resnet50(False)
        self.resnet.load_model('ckpts/resnet50-19c8e357.pth')
        self.resnet.eval()

        self.im_encoder = ImEncoder()
        self.depth_decoder = DepthDecoder()



    def forward(self, input):
        with torch.no_grad():
            fea1, fea2, fea3, fea4 = self.resnet.extract_feat(input)

        im_fea_list = self.im_encoder([fea4, fea3, fea2, fea1])
        pred = self.depth_decoder(im_fea_list)
        return pred.squeeze(1)


    def parameters(self):
        return  itertools.chain(
            self.depth_decoder.parameters(),
            self.im_encoder.parameters())

    def load_model(self, file_path):
        file_path = file_path.split('.pth')[0]
        self.depth_decoder.load_state_dict(
            torch.load(file_path+'_depth_decoder.pth'))
        self.im_encoder.load_state_dict(
            torch.load(file_path+'_im_encoder.pth'))
    
    def save_model(self, file_path):
        file_path = file_path.split('.pth')[0]
        self.cpu()
        torch.save(self.depth_decoder.state_dict(), 
            file_path+'_depth_decoder.pth')
        torch.save(self.im_encoder.state_dict(), 
            file_path+'_im_encoder.pth')
        self.cuda()
        

# class TempDepthNet(nn.Module):
#     def __init__(self, pretrain_ckpt, mode='joint'):
#         super(TempDepthNet, self).__init__()
#         self.mode = mode
#         self.resnet = resnet50(False)
#         self.resnet.load_model('ckpts/resnet50-19c8e357.pth')
#         self.resnet.eval()

#         self.im_encoder = ImEncoder()
#         self.depth_decoder = DepthDecoder()

#         file_path = pretrain_ckpt.split('.pth')[0]
#         self.depth_decoder.load_state_dict(
#             torch.load(file_path+'_depth_decoder.pth'))
#         self.im_encoder.load_state_dict(
#             torch.load(file_path+'_im_encoder.pth'))

#         self.flo_fuse_encoder = FloEarlyFusionEncoder(add_uv=False)
#         # self.flo_fuse_encoder = FloFusionEncoder()


#     def forward(self, im, flo):
#         with torch.no_grad():
#             fea1, fea2, fea3, fea4 = self.resnet.extract_feat(im)
#         im_fea_list = self.im_encoder([fea4, fea3, fea2, fea1])
#         fea_list = self.flo_fuse_encoder(flo, im_fea_list)
#         pred = self.depth_decoder(fea_list)
#         return pred.squeeze(1)


#     def parameters(self):
#         if self.mode == 'fix':
#             return  self.flo_fuse_encoder.parameters()
#         else:
#             return  itertools.chain(
#                 self.depth_decoder.parameters(),
#                 self.im_encoder.parameters(),
#                 self.flo_fuse_encoder.parameters())

#     def load_model(self, file_path):
#         if self.mode == 'fix':
#             self.flo_fuse_encoder.load_state_dict(
#                 torch.load(file_path))
#         else:
#             file_path = file_path.split('.pth')[0]
#             self.depth_decoder.load_state_dict(
#                 torch.load(file_path+'_depth_decoder.pth'))
#             self.im_encoder.load_state_dict(
#                 torch.load(file_path+'_im_encoder.pth'))
#             self.flo_fuse_encoder.load_state_dict(
#                 torch.load(file_path+'_flo_encoder.pth'))

    
#     def save_model(self, file_path):
#         self.cpu()
#         if self.mode == 'fix':
#             torch.save(self.flo_fuse_encoder.state_dict(), 
#                 file_path)
#         else:
#             file_path = file_path.split('.pth')[0]
#             torch.save(self.depth_decoder.state_dict(), 
#                 file_path+'_depth_decoder.pth')
#             torch.save(self.im_encoder.state_dict(), 
#                 file_path+'_im_encoder.pth')
#             torch.save(self.flo_fuse_encoder.state_dict(), 
#                 file_path+'_flo_encoder.pth')

#         self.cuda()




if __name__ == '__main__':
    # net = DepthNet()
    # pred = net.forward(torch.randn(1,3,384,384))
    # print(pred.size())
    net = TempDepthNet('/home/chaoyang/exp/yt3d_resnet_redweb_gradloss/model_cur')
    pred = net.forward(torch.randn(4,3,384,384),
        torch.randn(4,2,384,384))
    print(pred.size())
