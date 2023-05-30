# Copyright (c) OpenMMLab. All rights reserved.
from .gap_neck import GlobalAveragePooling
from .posewarper_neck import PoseWarperNeck
from .tcformer_mta_neck import MTA
__all__ = ['GlobalAveragePooling', 'PoseWarperNeck','MTA']
