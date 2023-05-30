_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/coco_fish.py'
]


checkpoint_config = dict(interval=20)
evaluation = dict(interval=20, metric='mAP', key_indicator='AP',save_best='AP')
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# optimizer = dict(
#     type='Adam',
#     lr=5e-4,)
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'relative_position_bias_table': dict(decay_mult=0.)}))

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 500#210
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])
max_num_people=2000
numKeyPoints = 1
channel_cfg = dict(
    num_output_channels=numKeyPoints,  #
    dataset_joints=numKeyPoints,  #
    dataset_channel=[
        list(range(numKeyPoints)),  #
    ],
    inference_channel=list(range(numKeyPoints)))
data_cfg = dict(
    image_size=1024,
    base_size=256,
    base_sigma=2,
    heatmap_size=[256],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=1,
    scale_aware_sigma=False,
    #################################
    # soft_nms=True,  # 推理过程中是否执行 soft_nms
    # nms_thr=0.3,  # 非极大抑制阈值
    # oks_thr=0.9,  # nms 期间 oks（对象关键点相似性）得分阈值
    # vis_thr=0.9,  # 关键点可见性阈值
)
# model settings
norm_cfg = dict(type='BN', requires_grad=True) #SyncBN
model = dict(
    type='SimpleEmbedding',
    pretrained='https://github.com/whai362/PVT/'
               'releases/download/v2/pvt_v2_b2.pth', #pvt_small
    backbone=dict(
        type='PyramidVisionTransformerV2', embed_dims=64, num_layers=[3, 4, 6, 3]),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=64,
        num_joints=channel_cfg['dataset_joints'],  #
        num_deconv_layers=0,
        tag_per_joint=False,  # True
        with_ae_loss=[False],
        extra=dict(final_conv_kernel=1, ),
        # loss_keypoint=dict(type='SingleLossFactory', use_target_weight=False)), #JointsMSELoss  True
        loss_keypoint=dict(type='SingleLossFactory',  # MultiLossFactory
                           num_joints=channel_cfg['dataset_joints'],  #
                           num_stages=1,
                           ae_loss_type='max',  # maX
                           with_ae_loss=[False],
                           push_loss_factor=[0.001],
                           pull_loss_factor=[0.001],
                           with_heatmaps_loss=[True],
                           heatmaps_loss_factor=[1.0],
                           supervise_empty=False)),
    # train_cfg=dict(),
    # test_cfg=dict(
    #     flip_test=True,
    #     post_process='default',
    #     shift_heatmap=True,
    #     modulate_kernel=11))
    train_cfg=dict(num_joints=channel_cfg['dataset_joints'],
                   img_size=data_cfg['image_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=max_num_people,
        scale_factor=[1],
        with_heatmaps=[True, ],
        with_ae=[False, ],  # True
        project2image=True,
        align_corners=False,
        nms_kernel=3,  # 5
        nms_padding=1,  # 2
        tag_per_joint=False,  # True
        detection_threshold=0.4,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=False,  # True
        flip_test=True))

# data_cfg = dict(
#     image_size=[192, 256],
#     heatmap_size=[48, 64],
#     num_output_channels=channel_cfg['num_output_channels'],
#     num_joints=channel_cfg['dataset_joints'],
#     dataset_channel=channel_cfg['dataset_channel'],
#     inference_channel=channel_cfg['inference_channel'],
#     soft_nms=False,
#     nms_thr=1.0,
#     oks_thr=0.9,
#     vis_thr=0.2,
#     use_gt_bbox=False,
#     det_bbox_thr=0.0,
#     bbox_file=f'{data_root}/person_detection_results/'
#               'COCO_val2017_detections_AP_H_56_person.json',
# )


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=max_num_people,
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1]),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline
data_root ='/home/chuanzhi/mnt_3T/lzc/waterflower/new'# '/home/chuanzhi/lzc/waterflower' /home/chuanzhi/mnt_3T/lzc/waterflower
fileDir = '10'#all
data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=4),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpFishDataset',
        ann_file=f'{data_root}/annotations/instances_train2022.json', # 80
        img_prefix=f'{data_root}/images/train2022',  #images/h  f'{data_root}/allImgs'
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpFishDataset',
        ann_file=f'{data_root}/annotations/instances_val2022.json', # val.json
        img_prefix=f'{data_root}/images/val2022', #/images/val
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    # test=dict(
    #     type='BottomUpCrowdPrawnDataset',
    #     ann_file=f'{data_root}/annotations/annotations_2kpt_test_4.json',
    #     img_prefix=f'{data_root}/images/',
    #     data_cfg=data_cfg,
    #     pipeline=test_pipeline,
    #     dataset_info={{_base_.dataset_info}}),

)
# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=2,
#     val_dataloader=dict(samples_per_gpu=32),
#     test_dataloader=dict(samples_per_gpu=32),
#     train=dict(
#         type='TopDownCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
#         img_prefix=f'{data_root}/train2017/',
#         data_cfg=data_cfg,
#         pipeline=train_pipeline,
#         dataset_info={{_base_.dataset_info}}),
#     val=dict(
#         type='TopDownCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         img_prefix=f'{data_root}/val2017/',
#         data_cfg=data_cfg,
#         pipeline=val_pipeline,
#         dataset_info={{_base_.dataset_info}}),
#     test=dict(
#         type='TopDownCocoDataset',
#         ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
#         img_prefix=f'{data_root}/val2017/',
#         data_cfg=data_cfg,
#         pipeline=val_pipeline,
#         dataset_info={{_base_.dataset_info}}),
# )

# fp16 settings
# fp16 = dict(loss_scale='dynamic')
