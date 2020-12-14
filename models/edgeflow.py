import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .mvsnet import MVSNet
from utils import *


class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()

        self.conv0 = ConvBnReLU(4, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 1, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv4 = ConvBnReLU(16, 32, 5, 1, 2)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(x)
        return x

class FeatureDown(nn.Module):
    def __init__(self, inplane):
        super(FeatureDown, self).__init__()
        self.down = nn.Sequential(
                nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
                nn.BatchNorm2d(inplane),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
                nn.BatchNorm2d(inplane),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        size = x.shape[-2], x.shape[-1]
        fea_down = self.down(x)
        fea_down = F.upsample(fea_down, size=size, mode="bilinear", align_corners=True)
        return torch.cat([x, fea_down], dim=1)

class EdgeFlow(nn.Module):
    def __init__(self, stages=[2, 4], grad_method="nodetach"):
        super(EdgeFlow, self).__init__()
        self.stages = [2, 4]
        self.grad_method = grad_method
        self.mvsnet = MVSNet()
        self.feature_exta = FeatureExtract()
        self.feature_down = FeatureDown(32)
        self.flow_make = nn.Conv2d(32*2, 2, kernel_size=3, padding=1, bias=False)
        # self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, imgs, proj_matrices, depth_values, threshold=5.0):
        # get init depth and confidence
        init_depth, con = self.mvsnet(imgs, proj_matrices, depth_values)

        depth = {"1x": init_depth}
        confidence = {"1x": con}
        edge = None
        # normal init depth
        scaled_depth = self.scale(init_depth, depth_values[:,0], depth_values[:,-1])

        curr_depth = scaled_depth
        for stage in self.stages:
            if self.grad_method == "detach":
                curr_depth = curr_depth.detach()
            curr_depth = F.upsample(curr_depth, scale_factor=2)
            curr_img = F.interpolate(imgs[:, 0, :, :], scale_factor=stage/4.0)

            # feature extract
            feature = self.feature_exta(torch.cat([curr_img, curr_depth], dim=1))

            # cat feature & upsample feature
            feature = self.feature_down(feature)

            # make flow 
            flow = self.flow_make(feature)

            depth_flow_warp = self.flow_warp(curr_depth, flow, curr_depth.size()[-2:])
            # depth_flow_warp = F.relu(self.conv(depth_flow_warp), inplace=True)
            if edge is None:
                edge = self.unscale(depth_flow_warp, depth_values[:,0], depth_values[:,-1]) - self.unscale(curr_depth, depth_values[:,0], depth_values[:,-1])
                edge = abs(F.interpolate(edge.unsqueeze(1), scale_factor=0.5).squeeze(1)) < threshold

            curr_depth = depth_flow_warp
            confidence["{}x".format(stage)] = F.upsample(con.unsqueeze(1), size=curr_depth.size()[-2:], mode="bilinear", align_corners=True).squeeze(1)
            depth["{}x".format(stage)] = self.unscale(depth_flow_warp, depth_values[:,0], depth_values[:,-1])

        return depth, confidence, edge.float()

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output
    
    def scale(self, depth, depth_min, depth_max):
        scaled_depth = (depth - depth_min) / (depth_max - depth_min)
        return scaled_depth.unsqueeze(1)
    def unscale(self, scaled_depth, depth_min, depth_max):
        depth = scaled_depth * (depth_max - depth_min) + depth_min
        return depth.squeeze(1)



def edgeflow_loss(_depth_est, _depth_gt, _mask, _edge=None):
    name_stages = ["1x", "2x", "4x"]
    weights = { "1x": 1.0,
                "2x": 1.0,
                "4x": 1.0}
    loss = {"total": 0.0}
    if _edge is not None:
        _mask['1x'] = _mask['1x'] * _edge
    for name in name_stages:
        mask, depth_est, depth_gt = _mask[name], _depth_est[name], _depth_gt[name]
        mask = mask > 0.5
        loss[name] = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        loss["total"] += loss[name] * weights[name]

    return loss
