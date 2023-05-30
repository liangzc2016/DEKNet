_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/coco_fish.py'
]
#for hrnet
checkpoint_config = dict(interval=20)
evaluation = dict(interval=20, metric='mAP', key_indicator='AP',save_best='AP')
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

optimizer = dict(
    type='Adam',
    lr=0.0015,#0.0015
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,  # 50
    warmup_ratio=0.001,  #0.001
    step=[20, 60]) #20,130
max_num_people=300
total_epochs = 500
numKeyPoints=1
channel_cfg = dict(
    num_output_channels=numKeyPoints,#
    dataset_joints=numKeyPoints,#
    dataset_channel=[
        list(range(numKeyPoints)),#
    ],
    inference_channel=list(range(numKeyPoints))) #

data_cfg = dict(
    image_size=256,
    base_size=256,
    base_sigma=2,
    heatmap_size=[64],
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
# pretrained = '/home/chuanzhi/lzc/mmpose-hrnet-test/tools/workdir/hrnet.pth',

model = dict(
    type='SimpleEmbedding', #AssociativeEmbedding
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w48-8ef0771d.pth',

    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),#4
                num_channels=(64,)),#64
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4), #4, 4
                num_channels=(48, 96)),#48,96 #32, 64
            stage3=dict(
                num_modules=4,#4
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4,4), #4, 4, 4
                num_channels=(48, 96, 192)),#48, 96, 192 #32, 64, 128
            stage4=dict(
                num_modules=3, #3
                num_branches=4, #4
                block='BASIC',
                num_blocks=(4, 4, 4, 4), #4, 4, 4, 4
                num_channels=(48, 96, 192, 384))),#48, 96, 192, 384  #32, 64, 128, 256
    ),
    keypoint_head=dict(
        type='AESimpleHead',
        in_channels=48, #48
        num_joints=channel_cfg['dataset_joints'], #
        num_deconv_layers=0,
        tag_per_joint=False,#True
        with_ae_loss=[False],
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(
            type='SingleLossFactory',#MultiLossFactory
            num_joints=channel_cfg['dataset_joints'], #
            num_stages=1,
            ae_loss_type='max',#maX
            with_ae_loss=[False],
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001],
            with_heatmaps_loss=[True],
            heatmaps_loss_factor=[1.0],
            supervise_empty=False)),

    train_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        img_size=data_cfg['image_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=max_num_people,
        scale_factor=[1],
        with_heatmaps=[True],
        with_ae=[False],
        project2image=True,
        align_corners=False,
        nms_kernel=5,#5
        nms_padding=2,#2
        tag_per_joint=False,
        detection_threshold=0.2,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=False,
        flip_test=True,
        # use_udp=True
    ))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='BottomUpRandomAffine',
        rot_factor=30,
        scale_factor=[0.75, 1.5],
        scale_type='short',
        trans_factor=40,
        # use_udp=True
    ),
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
        # use_udp=True,
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
        ],
        ),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]
test_pipeline = val_pipeline

data_root = '/home/chuanzhi/mnt_3T/lzc/cell/ad'
# data_root ='/home/chuanzhi/mnt_3T/lzc/duixia/same/one/head_tail/coco'# '/home/chuanzhi/lzc/waterflower' /home/chuanzhi/mnt_3T/lzc/waterflower
fileDir = '10'#all
data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=20),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='BottomUpFishDataset',
        ann_file=f'{data_root}/coco/train/train2023.json', # 80
        img_prefix=f'{data_root}/json',  #images/h  f'{data_root}/allImgs'
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='BottomUpFishDataset',
        ann_file=f'{data_root}/coco/val/val2023.json', # val.json
        img_prefix=f'{data_root}/json', #/images/val
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
