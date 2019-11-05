import torch
import numpy as np
from torch.autograd import Variable


def meshgrid(imH, imW):
    x = torch.from_numpy(np.arange(imW)).float()
    y = torch.from_numpy(np.arange(imH)).float()
    x = x.unsqueeze(0).repeat(imH, 1)
    y = y.unsqueeze(1).repeat(1, imW)
    return x, y


def grid_bilinear_sampling(img, x, y):
    batch_size, k, h, w = img.size()
    x_norm = x/((w-1)/2) - 1
    y_norm = y/((h-1)/2) - 1
    grid = torch.cat((x_norm.view(batch_size, h, w, 1),
                      y_norm.view(batch_size, h, w, 1)), 3)
    img_warped = torch.nn.functional.grid_sample(img,
                                                 grid, mode='bilinear', padding_mode='border')
    return img_warped.view(batch_size, k, h, w)


class Reproject3D(torch.nn.Module):
    def __init__(self, imH, imW, f):
        super(Reproject3D, self).__init__()
        u, v = meshgrid(imH, imW)
        self.register_buffer('u', u)
        self.register_buffer('v', v)
        x = (u-imW/2)/f
        y = (v-imH/2)/f
        self.imH = imH
        self.imW = imW
        self.f = f
        self.register_buffer('xy', torch.stack((x, y), 0))

    def forward(self, z, flo=None):
        z_ = z.unsqueeze(1)
        if flo is None:
            xyz = torch.cat((Variable(self.xy.unsqueeze(0))*z_,
                             z_), 1)
        else:
            u = Variable(self.u.unsqueeze(0))+flo[:, 0, ...]
            v = Variable(self.v.unsqueeze(0))+flo[:, 1, ...]
            z_ = grid_bilinear_sampling(z_, u, v)
            x = (u-self.imW/2)/self.f
            y = (v-self.imH/2)/self.f
            xy = torch.stack((x, y), 1)*z_
            xyz = torch.cat((xy, z_), 1)

        return xyz


class RodriguesLayer(torch.nn.Module):
    def __init__(self):
        super(RodriguesLayer, self).__init__()
        self.register_buffer('o', torch.zeros(1, 1))
        self.register_buffer('E', torch.eye(3))

    def skewsym(self, input):
        batch_size = input.size(0)
        o = Variable(self.o).expand(batch_size, 1)
        a0 = input[:, 0:1]
        a1 = input[:, 1:2]
        a2 = input[:, 2:3]
        return torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 1).view(batch_size, 3, 3)

    def forward(self, rot_vec):
        batch_size = rot_vec.size(0)
        rot_angle = rot_vec.norm(p=2, dim=1).view(
            batch_size, 1).clamp(min=1e-30)
        rot_axis = rot_vec / rot_angle
        A = self.skewsym(rot_axis)
        return Variable(self.E).view(1, 3, 3)\
            + A*rot_angle.sin().view(batch_size, 1, 1)\
            + A.bmm(A)*(1-rot_angle.cos()).view(batch_size, 1, 1)


class StereoWarp(torch.nn.Module):
    # x' = x+(cx'-cx)-bf/z
    def __init__(self, imH, imW):
        super(StereoWarp, self).__init__()
        x, y = meshgrid(imH, imW)
        self.imH = imH
        self.imW = imW
        self.register_buffer('x', x)
        self.register_buffer('y', y)

    def compute_x_warp(self, invdepth, bf, dcx):
        x = Variable(self.x.unsqueeze(0).expand_as(invdepth))
        return x - bf.view(-1, 1, 1)*invdepth + dcx.view(-1, 1, 1)

    def forward(self, img, invdepth, bf, dcx):
        x_warped = self.compute_x_warp(invdepth, bf, dcx)
        y = Variable(self.y.unsqueeze(0).expand_as(invdepth))
        mask = (x_warped >= 1) & (x_warped <= self.imW)
        img_warped = grid_bilinear_sampling(img, x_warped, y)
        return img_warped, mask.float()


class HorizontalWarp(StereoWarp):
    def __init__(self, imH, imW):
        super(HorizontalWarp, self).__init__(imH, imW)

    def forward(self, im, flo_x):
        x_warped = Variable(self.x.unsqueeze(0)) + flo_x
        y = Variable(self.y.unsqueeze(0).expand_as(flo_x))
        mask = (x_warped >= 1) & (x_warped <= self.imW)
        im_warped = grid_bilinear_sampling(im, x_warped, y)
        return im_warped, mask.float()


class FloWarp(torch.nn.Module):
    def __init__(self, imH, imW):
        super(FloWarp, self).__init__()
        x, y = meshgrid(imH, imW)
        self.imH = imH
        self.imW = imW
        self.register_buffer('x', x)
        self.register_buffer('y', y)

    def forward(self, im, flo_x, flo_y):
        x_warped = Variable(self.x.unsqueeze(0)) + flo_x
        y_warped = Variable(self.y.unsqueeze(0)) + flo_y
        im_warped = grid_bilinear_sampling(im, x_warped, y_warped)
        return im_warped
        # mask = mask & (u>0) & (u<(self.imW-2)) & (v>0) & (v<(self.imH-2))


class PerspectiveWarp(torch.nn.Module):
    def __init__(self, imH, imW):
        super(PerspectiveWarp, self).__init__()
        u, v = meshgrid(imH, imW)
        self.imH = imH
        self.imW = imW
        self.register_buffer('u', u)
        self.register_buffer('v', v)
        # self.register_buffer('e', torch.zeros(1))

    def extract_cam_vec(self, cam):
        fx = cam[:, 0]
        fy = cam[:, 1]
        cx = cam[:, 2]
        cy = cam[:, 3]
        return fx, fy, cx, cy

    def xy(self, cam):
        return self.uv2xy(cam,
                          Variable(self.u.unsqueeze(0)),
                          Variable(self.v.unsqueeze(0)))

    def xy2uv(self, cam, x, y):
        fx, fy, cx, cy = self.extract_cam_vec(cam)
        u = fx.view(-1, 1, 1)*x + cx.view(-1, 1, 1)
        v = fy.view(-1, 1, 1)*y + cy.view(-1, 1, 1)
        return u, v

    def uv2xy(self, cam, u, v):
        fx, fy, cx, cy = self.extract_cam_vec(cam)
        x = (u-cx.view(-1, 1, 1))/fx.view(-1, 1, 1)
        y = (v-cy.view(-1, 1, 1))/fy.view(-1, 1, 1)
        return x, y

    def reproject(self, cam1, cam2, d1, rotm, t):
        batch_size = d1.size(0)
        x, y = self.xy(cam1)
        xyz = rotm[:, :, 0:1]*x.view(1, 1, -1) \
            + rotm[:, :, 1:2]*y.view(1, 1, -1) \
            + rotm[:, :, 2:3] \
            + d1.view(batch_size, 1, -1)*t.unsqueeze(-1)
        z = xyz[:, 2:3, :]
        mask = (z > 1e-5).view_as(d1)
        xy = xyz[:, 0:2, :] / z
        x = xy[:, 0, :].view_as(d1)
        y = xy[:, 1, :].view_as(d1)
        u, v = self.xy2uv(cam2, x, y)
        mask = mask & (u > 0) & (u < (self.imW-2)
                                 ) & (v > 0) & (v < (self.imH-2))
        return u, v, mask.view_as(d1)

    def forward(self, cam1, cam2, im2, d1, rotm, t):
        u, v, mask = self.reproject(cam1, cam2, d1, rotm, t)
        im2_warped = grid_bilinear_sampling(im2, u, v)
        return im2_warped, mask.float()
