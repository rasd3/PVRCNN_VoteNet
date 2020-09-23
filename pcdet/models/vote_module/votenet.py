# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys

import numpy as np

import torch
import torch.nn as nn
from .backbone_module import Pointnet2Backbone
from .voting_module import VotingModule

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, input_feature_dim=0, vote_factor=1):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.vote_factor = vote_factor

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)


    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs.shape[0]

        end_points = self.backbone_net(inputs, end_points)
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']

        vote_xyz, offset = self.vgen(xyz, features)

        return vote_xyz, offset.squeeze()


if __name__ == '__main__':
    inp = torch.zeros((4, 2048, 3)).cuda()
    votenet = VoteNet(0).cuda()
    import pdb; pdb.set_trace()
    abc = votenet(inp)
    abde = 1

        
