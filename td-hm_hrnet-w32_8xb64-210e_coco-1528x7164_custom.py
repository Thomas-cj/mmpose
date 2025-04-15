# https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf


_base_ = ['configs/_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=20, val_interval=4)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
"""
LinearLR: A linear warm-up phase where the learning rate increases linearly 
    from 0.0005 * 0.001 to 0.0005 over the first 500 iterations (not epochs).

MultiStepLR: After the warm-up, the learning rate is decreased by a factor of 0.1 
    at epochs 170 and 200. This is a common technique to fine-tune the model in later stages.
"""
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=5, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=5,
        milestones=[10, 15],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1)

# hooks
"""
This sets up a checkpoint hook to save the best model based on the Average Precision (AP)
metric on the COCO validation dataset.
"""
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
"""
codec = dict(...): This defines the codec settings for converting keypoint coordinates 
    to heatmaps and vice versa.

type='MSRAHeatmap': The MSRA heatmap generation method is used.
input_size=(288, 384): The input image size for the model. (Width, Height)
heatmap_size=(72, 96): The size of the generated heatmaps.
sigma=3: The standard deviation of the Gaussian kernel used to generate heatmaps.

"""
# codec = dict(
#     type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)

codec = dict(
    type='MSRAHeatmap', input_size=(7164,1528), heatmap_size=(1791, 382), sigma=3)

# model settings
# Need to edit for dataset mean and std
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean= [25.68008436, 74.64080128, 76.65044621],
        std= [41.30127197, 84.055176,   84.67723933],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=15,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/SwimDK_small/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    #dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/SwimDK_small_train.json',
        data_prefix=dict(img='SwimDK_small_train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/SwimDK_small_val.json',
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='SwimDK_small_val/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/SwimDK_small_val.json')
test_evaluator = val_evaluator
