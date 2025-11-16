import time

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from .instance_extraction import TransFusionHeadV2

assert 0, -1


@MODELS.register_module()
class InstanceFusion(nn.Module):
    def __init__(
        self,
        num_proposals=200,
        auxiliary=True,
        in_channels=256 * 2,
        hidden_channel=128,
        num_classes=10,
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0,
        ),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        bev_shape=[180, 180],
        point_cloud_range=[-54, -54, -5, 54, 54, 3],
        voxel_size=[0.075, 0.075, 0.2],
        out_size_factor=8,
    ):
        super(InstanceFusion, self).__init__()
        self.ins_ext = TransFusionHeadV2(
            num_proposals=num_proposals,
            auxiliary=auxiliary,
            in_channels=in_channels,
            hidden_channel=hidden_channel,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            nms_kernel_size=nms_kernel_size,
            ffn_channel=ffn_channel,
            dropout=dropout,
            bn_momentum=bn_momentum,
            activation=activation,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_heatmap=loss_heatmap,
            bev_shape=bev_shape,
            voxel_size=voxel_size,
            pc_range=point_cloud_range,
            out_size_factor=out_size_factor,
        )

    def forward(self, input):
        ins = [self.ins_ext(tmp)[0][0] for tmp in input if tmp is not None]

        return ins

    # def loss(self, input, target):
