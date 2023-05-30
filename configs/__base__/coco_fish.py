dataset_info = dict(
    # dataset_name='BottomUpFishDataset',
    dataset_name='BottomUpFishDataset',
    paper_info=dict(
        author='Jin, Sheng and Xu, Lumin and Xu, Jin and '
        'Wang, Can and Liu, Wentao and '
        'Qian, Chen and Ouyang, Wanli and Luo, Ping',
        title='Whole-Body Human Pose Estimation in the Wild',
        container='Proceedings of the European '
        'Conference on Computer Vision (ECCV)',
        year='2020',
        homepage='https://github.com/jin-s13/COCO-WholeBody/',
    ),
    keypoint_info={
        0:
        dict(name='head', id=0, color=[0, 255, 0], type='', swap=''),
        # 1:
        # dict(name='abdomen', id=1, color=[0, 255, 255], type='', swap='')

    },
    skeleton_info={
        0:
        dict(link=('head', 'head'), id=0, color=[255, 255, 0]) #abdomen
    },
    joint_weights=[1.],
    # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
    # 'evaluation/myeval_wholebody.py#L175'
    sigmas=[
        1 #1 #0.05 5  1 8 12 3
    ])
