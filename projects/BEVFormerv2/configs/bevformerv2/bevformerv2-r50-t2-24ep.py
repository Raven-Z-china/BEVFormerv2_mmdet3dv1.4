# mAP: 0.4600
# mATE: 0.6185
# mASE: 0.2815
# mAOE: 0.3660
# mAVE: 0.3157
# mAAE: 0.1902
# NDS: 0.5528
_base_ = ['../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=[
        'projects.BEVFormerv2.bevformerv2',
    ],
    allow_failed_imports=False,
)

total_epochs = 24

# Dataset
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
]
dataset_type = 'CustomNuScenesDatasetV2'
data_root = 'data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)
img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1, 1, 1], to_rgb=False)
bev_h_ = 200
bev_w_ = 200
frames = (-1, 0)
group_detr = 11
voxel_size = [102.4 / bev_h_, 102.4 / bev_w_, 8]
out_size_factor = 4
ida_aug_conf = {
    'reisze': [512, 544, 576, 608, 640, 672, 704, 736, 768],  #  (0.8, 1.2)
    'crop': (0, 260, 1600, 900),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}
ida_aug_conf_eval = {
    'reisze': [
        640,
    ],
    'crop': (0, 260, 1600, 900),
    'H': 900,
    'W': 1600,
    'rand_flip': False,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='GlobalRotScaleTransImage',
        rot_range=[-22.5, 22.5],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=True,
        training=True,
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
        only_gt=True,),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=class_names),
    dict(type='CropResizeFlipImage', data_aug_conf=ida_aug_conf, training=True, debug=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=['gt_bboxes_3d', 'gt_labels_3d', 'img',
              'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation',
              'timestamp', 'mono_input_dict', 'mono_ann_idx', 'aug_param']),
    dict(type='DD3DMapper',
         is_train=True,
         tasks=dict(box2d_on=True, box3d_on=True),)
]
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6,test_mode=True),
    dict(type='CropResizeFlipImage', data_aug_conf=ida_aug_conf_eval, training=False, debug=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 640),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D',
                 keys=['img', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation',
                       'lidar2ego_rotation', 'timestamp'])
        ])
]


train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CustomNuScenesDatasetV2',
        frames=frames,
        data_root=data_root,
        ann_file='nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        serialize_data=False,
        metainfo=dict(classes=class_names,version='v1.0-mini'),
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        mono_cfg=dict(
            name='nusc_mini_train',
            data_root=data_root,
            min_num_lidar_points=3,
            min_box_visibility=0.2))
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CustomNuScenesDatasetV2',
        frames=frames,
        data_root=data_root,
        serialize_data=False,
        ann_file='nuscenes_infos_temporal_val.pkl',
        metainfo=dict(classes=class_names,version='v1.0-mini'),
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,)
)

test_dataloader = val_dataloader


val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root+'nuscenes_infos_temporal_val.pkl',
    metric='bbox',
)

test_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root+'nuscenes_infos_temporal_val.pkl',
    metric='bbox',
)


train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_begin=1, val_interval=4
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# model
load_from = './ckpts/fcos_r50_coco_2mmdet.pth'
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
_num_mono_levels_ = 5

model = dict(
    type='BEVFormerV2',
    use_grid_mask=True,
    video_test_mode=False,
    num_levels=_num_levels_,
    num_mono_levels=_num_mono_levels_,
    mono_loss_weight=1.0,
    frames=frames,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN'),
        norm_eval=False,
        style='caffe',
    ),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_mono_levels_,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type='BEVFormerHead_GroupDETR',
        group_detr=group_detr,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformerV2',
            embed_dims=_dim_,
            frames=frames,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=4,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                ),
            ),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='GroupMultiheadAttention',
                            group=group_detr,
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=0.1,
                    ),
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type='mmdet.LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
    ),
    fcos3d_bbox_head=dict(
        type='NuscenesDD3D',
        num_classes=10,
        in_channels=_dim_,
        strides=[8, 16, 32, 64, 128],
        box3d_on=True,
        feature_locations_offset='none',
        fcos2d_cfg=dict(
            num_cls_convs=4,
            num_box_convs=4,
            norm='SyncBN',
            use_deformable=False,
            use_scale=True,
            box2d_scale_init_factor=1.0,
        ),
        fcos2d_loss_cfg=dict(
            focal_loss_alpha=0.25, focal_loss_gamma=2.0, loc_loss_type='giou'
        ),
        fcos3d_cfg=dict(
            num_convs=4,
            norm='SyncBN',
            use_scale=True,
            depth_scale_init_factor=0.3,
            proj_ctr_scale_init_factor=1.0,
            use_per_level_predictors=False,
            class_agnostic=False,
            use_deformable=False,
            mean_depth_per_level=[44.921, 20.252, 11.712, 7.166, 8.548],
            std_depth_per_level=[24.331, 9.833, 6.223, 4.611, 8.275],
        ),
        fcos3d_loss_cfg=dict(
            min_depth=0.1,
            max_depth=80.0,
            box3d_loss_weight=2.0,
            conf3d_loss_weight=1.0,
            conf_3d_temperature=1.0,
            smooth_l1_loss_beta=0.05,
            max_loss_per_group=20,
            predict_allocentric_rot=True,
            scale_depth_by_focal_lengths=True,
            scale_depth_by_focal_lengths_factor=500.0,
            class_agnostic=False,
            predict_distance=False,
            canon_box_sizes=[
                [2.3524184, 0.5062202, 1.0413622],
                [0.61416006, 1.7016163, 1.3054738],
                [2.9139307, 10.725025, 3.2832346],
                [1.9751819, 4.641267, 1.74352],
                [2.772134, 6.565072, 3.2474296],
                [0.7800532, 2.138673, 1.4437162],
                [0.6667362, 0.7181772, 1.7616143],
                [0.40246472, 0.4027083, 1.0084083],
                [3.0059454, 12.8197, 4.1213827],
                [2.4986045, 6.9310856, 2.8382742],
            ],
        ),
        target_assign_cfg=dict(
            center_sample=True,
            pos_radius=1.5,
            sizes_of_interest=(
                (-1, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, 100000000.0),
            ),
        ),
        nusc_loss_weight=dict(attr_loss_weight=0.2, speed_loss_weight=0.2),
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),  # 0.75
                iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=4e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.5, decay_mult=0.0),
            transformer=dict(lr_mult=1.5),
            reference_points=dict(lr_mult=2.0, decay_mult=0.0),
            query_embed=dict(lr_mult=2.0, decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            bias=dict(decay_mult=0.0),
        )
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        begin=0,
        end=1500,
        by_epoch=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=2e-6,
        by_epoch=True,
        begin=0,
        end=24,
    ),
]

fp16 = dict(
    loss_scale=512.0,
    loss_scale_mode='dynamic',
    init_scale=2.0**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)

model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True)
