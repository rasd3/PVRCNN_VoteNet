#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import datetime
import glob
import os

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pcdet.config import (cfg, cfg_from_list, cfg_from_yaml_file,
                          log_config_to_file)
from pcdet.datasets import build_dataloader
from pcdet.models import model_fn_decorator
from pcdet.models.vote_module.votenet import VoteNet
from pcdet.ops.pointnet2.pointnet2_stack import \
    pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.pointnet2.pointnet2_stack import \
    pointnet2_utils as pointnet2_stack_utils
from pcdet.utils import common_utils
from tensorboardX import SummaryWriter
from tools.train import parse_config
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def make_match_diff(gt_boxes, keypoints, NUM_KEYPOINTS, ret_closest=False):
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
    match_diff = closest_box - keypoints
    if False:
        # For utility test (confirm)
        match_diff_norm = torch.gather(diff, 0, diff_ind.unsqueeze(1)).sqrt()
        match_diff /= 5.
        keypoints += match_diff
    match_diff = match_diff.unsqueeze(dim=0)

    if ret_closest:
        return match_diff, closest_box
    else:
        return match_diff


class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = 0

        if self.model_cfg.SAMPLE_METHOD_VOTE.USE_VOTE:
            self.losses_cfg = self.model_cfg.SAMPLE_METHOD_VOTE.LOSS_WEIGHTS
            self.key_votenet = VoteNet(0)
            self.forward_vote_dict = {}
            self.reg_loss_func = F.smooth_l1_loss


    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        vsa_loss, tb_dict_1 = self.get_vote_reg_layer_loss()
        tb_dict.update(tb_dict_1)

        return vsa_loss, tb_dict

    def get_vote_reg_layer_loss(self, tb_dict=None):
        vote_preds = self.forward_vote_dict['preds']
        vote_gt = self.forward_vote_dict['gt']
        batch_size = int(vote_gt.shape[0])

        reg_loss_src = self.reg_loss_func(vote_preds, vote_gt)
        #  reg_loss_src = torch.div(torch.sum(torch.abs(vote_gt - vote_preds)), vote_gt.shape[1])
        reg_loss = reg_loss_src.sum() / batch_size
        reg_loss = reg_loss * self.losses_cfg['reg_weight']

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vsa_loss': reg_loss.item()})

        vsa_loss = reg_loss

        return vsa_loss, tb_dict


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

                if self.model_cfg.SAMPLE_METHOD_VOTE.USE_VOTE:
                    # GT offset for each keypoint
                    match_diff, closest_box = make_match_diff(batch_dict['gt_boxes'][bs_idx],
                                                              keypoints,
                                                              self.model_cfg.NUM_KEYPOINTS,
                                                              ret_closest=True)
                    #  keypoints_diff_list.append(match_diff)
                    keypoints_diff_list.append(closest_box.unsqueeze(0))

                keypoints = keypoints.unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'Vote':
                point_cloud, choices = random_sampling(sampled_points.squeeze(), 26000, return_choices=True)
                keypoints = point_cloud.unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError

            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        if self.model_cfg.SAMPLE_METHOD == 'FPS' and \
                self.model_cfg.SAMPLE_METHOD_VOTE.USE_VOTE:
            keypoints_dbg = keypoints.clone()
            keypoints_diff = torch.cat(keypoints_diff_list, dim=0)
            keypoints_diff = torch.tensor(keypoints_diff, requires_grad=True)
            keypoints, offset = self.key_votenet(keypoints)
            #  self.forward_vote_dict['preds'] = offset
            self.forward_vote_dict['gt'] = keypoints_diff
            self.forward_vote_dict['preds'] = keypoints

        if self.model_cfg.SAMPLE_METHOD == 'Vote':
            keypoints, offset = self.key_votenet(keypoints)
            for bs_idx in range(batch_size):
                match_diff, closest_box = make_match_diff(batch_dict['gt_boxes'][bs_idx],
                                             keypoints[bs_idx], self.model_cfg.NUM_KEYPOINTS,
                                             ret_closest=True)
                keypoints_diff_list.append(closest_box.unsqueeze(0))
            keypoints_diff = torch.cat(keypoints_diff_list, dim=0)
            self.forward_vote_dict['gt'] = keypoints_diff
            self.forward_vote_dict['preds'] = keypoints

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

        return batch_dict
        

class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'pfe'
        ]

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = VoxelSetAbstraction(
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict, tb_dict = {}, {}
        loss_pfe, tb_dict = self.pfe.get_loss(tb_dict)
        loss = loss_pfe

        return loss, tb_dict, disp_dict


def main():
    args, cfg = parse_config()
    if args.debug:
        args.workers = 0

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    model = PVRCNN(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
    )
    
    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()
