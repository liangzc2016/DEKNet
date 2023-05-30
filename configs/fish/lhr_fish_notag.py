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

optimizer = dict(
    type='Adam',
    lr=0.015,#0.0015
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,  # 50
    warmup_ratio=0.001,  #0.001
    step=[20, 60]) #20,130  20, 60

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
    image_size=1024, #512 640 1024
    base_size=256,
    base_sigma=2,
    heatmap_size=[1024,], #256  512
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
max_num_people=2000
num_joints=1
model = dict(
    type='SimpleEmbedding', #AssociativeEmbedding SimpleEmbedding
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
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=1,#4
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96),
                ),
            stage4=dict(
                num_modules=1,#3
                num_branches=2,#4
                block='BASIC',
                num_blocks=(4 ,4,),  #4, 4,4,4
                num_channels=(48,96), #48,96, 192,384
                )   #multiscale_output=True 获取所有尺度
                ),
    ),
    keypoint_head=dict(
        type='AEHigherResolutionHead3',
        in_channels=[48,96,],
        num_joints=num_joints,
        tag_per_joint=False,#True
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=1,#1
        num_deconv_filters=[48,96,],
        num_deconv_kernels=[4,4,],
        num_basic_blocks=0, #0
        cat_output=[False,False,],
        with_ae_loss=[False,False,False],
        needDW=False, #False
        need2heatmap=True,
        otherLoss=False,
        loss_keypoint=dict(
            type='SingleLossFactory', #MultiLossFactory SingleLossFactory
            num_joints=num_joints,
            num_stages=1,
            ae_loss_type='exp',
            with_ae_loss=[False,], #True
            push_loss_factor=[0.001],
            pull_loss_factor=[0.001, ],
            with_heatmaps_loss=[True, ],
            heatmaps_loss_factor=[1.0,])),#1.2
    train_cfg=dict(num_joints=channel_cfg['dataset_joints'],
        img_size=data_cfg['image_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=max_num_people,
        scale_factor=[1],
        with_heatmaps=[True,],
        with_ae=[False,],#True
        project2image=True,
        align_corners=False,
        nms_kernel=3,#5
        nms_padding=1,#2
        tag_per_joint=False, #True
        detection_threshold=0.4,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True, #True
        refine=False, #True
        flip_test=True))

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
