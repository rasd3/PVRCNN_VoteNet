import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....ops.pointnet2.pointnet2_stack import \
    pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import \
    pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
from ...vote_module.votenet import VoteNet


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def make_match_diff(gt_boxes, keypoints, NUM_KEYPOINTS):
    # GT offset for each keypoint
    NUM_GT_BOX = gt_boxes.shape[0]
    gt_box = gt_boxes[:, :3]
    zero_mask = gt_box.pow(2).sum(1) == 0
    gt_box[zero_mask] = torch.empty(gt_box[zero_mask].shape).fill_(9999).cuda()
    gt_box_diff = gt_box.unsqueeze(1).repeat(1, NUM_KEYPOINTS, 1)
    keyp_diff = keypoints.unsqueeze(0).repeat(NUM_GT_BOX, 1, 1)
    diff = (gt_box_diff - keyp_diff).pow(2).sum(2)
    diff_ind = diff.min(0)[1]
    closest_box = gt_box[diff_ind][:, :3]
    if False:
        # For utility test (confirm)
        match_diff_norm = torch.gather(diff, 0, diff_ind.unsqueeze(1)).sqrt()
        match_diff /= 5.
        keypoints += match_diff
        match_diff = closest_box - keypoints
        match_diff = match_diff.unsqueeze(dim=0)

    return closest_box #, match_diff.squeeze().pow(2).sum(1).sqrt().mean()

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]

            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

        self.forward_vote_dict = {}

        if self.model_cfg.SAMPLE_METHOD_VOTE.USE_VOTE_LOSS or\
                self.model_cfg.SAMPLE_METHOD == 'Vote':
            self.losses_cfg = self.model_cfg.SAMPLE_METHOD_VOTE.LOSS_WEIGHTS
            self.key_votenet = VoteNet(0)
            #  self.reg_loss_func = F.smooth_l1_loss


    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        vsa_loss, tb_dict_1 = self.get_vote_reg_layer_loss()
        tb_dict.update(tb_dict_1)

        return vsa_loss, tb_dict

    def get_vote_reg_layer_loss(self, tb_dict=None):
        vote_preds = self.forward_vote_dict['preds']
        vote_gt = self.forward_vote_dict['gt']
        batch_size = int(vote_gt.shape[0])

        #  reg_loss_src = self.reg_loss_func(vote_preds, vote_gt)
        reg_loss = torch.div(torch.sum(torch.abs(vote_gt - vote_preds)), vote_gt.shape[1])
        reg_loss = reg_loss * self.losses_cfg['reg_weight']

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vsa_loss': reg_loss.item()})

        return reg_loss, tb_dict

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        keypoints_diff_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]]

                if self.model_cfg.SAMPLE_METHOD_VOTE.USE_VOTE_LOSS:
                    # GT offset for each keypoint
                    #  closest_box = make_match_diff(batch_dict['gt_boxes'][bs_idx],
                                                  #  keypoints,
                                                  #  self.model_cfg.NUM_KEYPOINTS)
                    #  keypoints_diff_list.append(match_diff)
                    #  keypoints_diff_list.append(closest_box.unsqueeze(0))
                    pass

                keypoints = keypoints.unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'Vote':
                point_cloud = random_sampling(sampled_points.squeeze(), 25000)
                keypoints = point_cloud.unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError

            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        #  keypoints.requires_grad = True
        if self.model_cfg.SAMPLE_METHOD == 'FPS' and \
                self.model_cfg.SAMPLE_METHOD_VOTE.USE_VOTE_LOSS:
            #  keypoints_diff = torch.cat(keypoints_diff_list, dim=0)
            #  keypoints = self.key_votenet(keypoints)
            #  self.forward_vote_dict['preds'] = keypoints
            #  self.forward_vote_dict['gt'] = keypoints_diff
            pass

        if self.model_cfg.SAMPLE_METHOD == 'Vote':
            #  keypoints = self.key_votenet(keypoints)
            for bs_idx in range(batch_size):
                closest_box = make_match_diff(batch_dict['gt_boxes'][bs_idx],
                                              keypoints[bs_idx],
                                              self.model_cfg.NUM_KEYPOINTS)
                keypoints_diff_list.append(closest_box.unsqueeze(0))

            keypoints_diff = torch.cat(keypoints_diff_list, dim=0)
            self.forward_vote_dict['preds'] = keypoints
            self.forward_vote_dict['gt'] = keypoints_diff

        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)

        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
            point_features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

            pooled_points, pooled_features = self.SA_rawpoints(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features,
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=2)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict
