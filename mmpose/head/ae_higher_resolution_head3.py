# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init, build_activation_layer, build_norm_layer)
from mmcv.runner import BaseModule

from mmpose.models.builder import build_loss
from ..backbones.resnet import BasicBlock
from ..builder import HEADS

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()
@HEADS.register_module()
class AEHigherResolutionHead3(nn.Module):
    """Associative embedding with higher resolution head. paper ref: Bowen
    Cheng et al. "HigherHRNet: Scale-Aware Representation Learning for Bottom-
    Up Human Pose Estimation".

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        extra (dict): Configs for extra conv layers. Default: None
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        cat_output (list[bool]): Option to concat outputs.
        with_ae_loss (list[bool]): Option to use ae loss.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 tag_per_joint=True,
                 extra=None,
                 num_deconv_layers=1,
                 num_deconv_filters=(32, ),
                 num_deconv_kernels=(4, ),
                 num_basic_blocks=4,
                 cat_output=None,
                 with_ae_loss=None,
                 loss_keypoint=None,
                 needDW=True,
                 need2heatmap=False,
                 otherLoss = False):
        super().__init__()

        self.loss = build_loss(loss_keypoint)
        dim_tag = num_joints if tag_per_joint else 1

        self.num_deconvs = num_deconv_layers
        self.cat_output = cat_output

        final_layer_output_channels = []

        if with_ae_loss[0]:
            out_channels = num_joints + dim_tag
        else:
            out_channels = num_joints

        final_layer_output_channels.append(out_channels)
        for i in range(num_deconv_layers):
            if with_ae_loss[i + 1]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            final_layer_output_channels.append(out_channels)

        deconv_layer_output_channels = []
        for i in range(num_deconv_layers):
            if with_ae_loss[i]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            deconv_layer_output_channels.append(out_channels)
        self.need2heatmap = need2heatmap  # True，再反卷积到原图大小，False则原图的1/2
        self.final_layers = self._make_final_layers(
            in_channels, final_layer_output_channels, extra, num_deconv_layers,
            num_deconv_filters)
        self.deconv_layers = self._make_deconv_layers(
            in_channels, deconv_layer_output_channels, num_deconv_layers,
            num_deconv_filters, num_deconv_kernels, num_basic_blocks,
            cat_output,need2heatmap)
        # 以下会占用参数的大小
        # self.deconvFirst = self.deconv_first(in_channels,
        #                      num_deconv_filters,
        #                     num_deconv_kernels,cat_output)
        #以下会占用参数的大小
        # self.ffn = CrossFFN(in_channels[0],hidden_features=int(in_channels[0] * 4),
        #     out_features=in_channels[0])
        # self.ffn2 = CrossFFN(in_channels[1], hidden_features=int(in_channels[1] * 4),
        #                     out_features=in_channels[1])
        # self.norm2 = build_norm_layer(dict(type='LN', eps=1e-6), in_channels[0])[1]
        # self.norm3 = build_norm_layer(dict(type='LN', eps=1e-6), in_channels[1])[1]
        self.needDW = needDW  #默认True添加深度卷积，False不需要添加深度卷积
        self.otherLoss = otherLoss


    @staticmethod
    def _make_final_layers(in_channels, final_layer_output_channels, extra,
                           num_deconv_layers, num_deconv_filters):
        """Make final layers."""
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            else:
                padding = 0
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        final_layers = []
        final_layers.append(
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                in_channels=in_channels[0],
                out_channels=final_layer_output_channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=padding))

        # for i in range(num_deconv_layers):
        #     in_channels = num_deconv_filters[i]
        #     final_layers.append(
        #         build_conv_layer(
        #             cfg=dict(type='Conv2d'),
        #             in_channels=in_channels,
        #             out_channels=final_layer_output_channels[i + 1],
        #             kernel_size=kernel_size,
        #             stride=1,
        #             padding=padding))

        return nn.ModuleList(final_layers)

    def deconv_first(self, in_channels,
                             num_deconv_filters,
                            num_deconv_kernels,cat_output):
        planes = num_deconv_filters[0]
        deconv_kernel, padding, output_padding = \
            self._get_deconv_cfg(num_deconv_kernels[0])
        layers = []
        if cat_output:
            layers.append(
                nn.Sequential(
                    build_upsample_layer(
                        dict(type='deconv'),
                        in_channels=(in_channels[0]+2),
                        out_channels=planes,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False), nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True)))
            return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels, deconv_layer_output_channels,
                            num_deconv_layers, num_deconv_filters,
                            num_deconv_kernels, num_basic_blocks, cat_output,need2heatmap):
        """Make deconv layers."""
        deconv_layers = []
        for i in range(num_deconv_layers):
            # if cat_output[i]:
            #     in_channels[i] += deconv_layer_output_channels[i]

            planes = num_deconv_filters[0]
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(num_deconv_kernels[i])
            # in_channels[i]=(int)(num_deconv_filters[0]*(i+1))
            layers = []
            for j in range(i+1):
                # print('in_channels[i] // (2 ** j)',in_channels[i],2**j,in_channels[i] // (2 ** j))
                if j==i:
                    layers.append(
                        nn.Sequential(
                            build_upsample_layer(
                                dict(type='deconv'),
                                in_channels=(in_channels[i]//(2**j)),
                                out_channels=planes,
                                kernel_size=deconv_kernel,
                                stride=2, # 2  #4
                                padding=padding, #padding #0
                                output_padding=output_padding, #output_padding
                                bias=False), nn.BatchNorm2d(planes, momentum=0.1),
                            nn.ReLU(inplace=True)))
                    if need2heatmap:
                        layers.append(
                            nn.Sequential(
                                build_upsample_layer(
                                    dict(type='deconv'),
                                    in_channels=(in_channels[i] // (2 ** j)),
                                    out_channels=planes,
                                    kernel_size=deconv_kernel,
                                    stride=2,
                                    padding=padding,
                                    output_padding=output_padding,
                                    bias=False), nn.BatchNorm2d(planes, momentum=0.1),
                                nn.ReLU(inplace=True)))
                # else:
                #     inC = in_channels[i] // (2**j)
                #     layers.append(
                #         nn.Sequential(
                #             build_upsample_layer(
                #                 dict(type='deconv'),
                #                 in_channels=(inC),#从高维反向卷积，如96-》48
                #                 out_channels=inC//2,
                #                 kernel_size=deconv_kernel,
                #                 stride=2,
                #                 padding=padding,
                #                 output_padding=output_padding,
                #                 bias=False), nn.BatchNorm2d(inC//2, momentum=0.1),
                #             nn.ReLU()))
            # for _ in range(num_basic_blocks):
            #     layers.append(nn.Sequential(BasicBlock(planes, planes), ))
            deconv_layers.append(nn.Sequential(*layers))
            # in_channels = planes

        return nn.ModuleList(deconv_layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def get_loss(self, outputs, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (list(torch.Tensor[N,K,H,W])): Multi-scale output heatmaps.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints (List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
        """

        losses = dict()
        if self.otherLoss:
            heatmaps_losses = self.loss(outputs, targets, 1)
        else:
            heatmaps_losses, push_losses, pull_losses = self.loss(
             outputs, targets, masks, joints)


        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            # if push_losses[idx] is not None:
            #     push_loss = push_losses[idx].mean(dim=0)
            #     if 'push_loss' not in losses:
            #         losses['push_loss'] = push_loss
            #     else:
            #         losses['push_loss'] += push_loss
            # if pull_losses[idx] is not None:
            #     pull_loss = pull_losses[idx].mean(dim=0)
            #     if 'pull_loss' not in losses:
            #         losses['pull_loss'] = pull_loss
            #     else:
            #         losses['pull_loss'] += pull_loss
        # print('heatmap_loss: ',losses['heatmap_loss'].cpu().detach().numpy())
        return losses

    def performDW(self, temp,flag=0):
        B, C, H, W = temp.size()
        temp = temp.view(B, C, -1).permute(0, 2, 1)
        # FFN
        # if flag == 0:
        #     temp = self.ffn(self.norm2(temp), H, W)
        # elif flag == 1:
        #     temp = self.ffn2(self.norm3(temp), H, W)
        # temp = self.ffn(self.norm2(temp), H, W) #再加一个个深度卷积
        temp = temp.permute(0, 2, 1).view(B, C, H, W)
        return temp

    def forward(self, x):
        """Forward function."""
        # if isinstance(x, list):
        #     # x = x[0]
        #     for i in range(len(x)):
        #         print('x[i].shape',i,'  ',x[i].shape)

        final_outputs = []
        y = self.final_layers[0](x[0])
        if self.cat_output[0]:
            x[0] = torch.cat((x[0], y), 1)
        # final_outputs.append(y)
        # x[0]=self.performDW(x[0],0)
        # x[1]=self.performDW(x[1],1)
        for i in range(self.num_deconvs):
            # if self.cat_output[0] and i==0:
            #     xx = self.deconvFirst(x[0])
            # else:
            #     xx = self.deconv_layers[i](x[i])
            xx = self.deconv_layers[i](x[i])
            # print('x[i].shape ',x[i].shape)
            # xx = self.deconv_layers[i](x[i])
            # print('x.size()==',xx.size())
            if i==0:
                temp=xx
            else:
                temp=temp+xx  # temp+=xx会出错
            # print('temp.size()==',temp.size())
        if self.needDW:
            temp=self.performDW(temp)

        y = self.final_layers[0](temp)
        final_outputs.append(y)
        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for _, m in self.final_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

class CrossFFN(BaseModule):
    r"""FFN with Depthwise Conv of HRFormer.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int, optional): The hidden dimension of FFNs.
            Defaults: The same as in_features.
        act_cfg (dict, optional): Config of activation layer.
            Default: dict(type='GELU').
        dw_act_cfg (dict, optional): Config of activation layer appended
            right after DW Conv. Default: dict(type='GELU').
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='SyncBN').
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 dw_act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'), #多卡才用SyncBN
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = build_activation_layer(act_cfg)
        self.norm1 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1)
        self.act2 = build_activation_layer(dw_act_cfg)
        self.norm2 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = build_activation_layer(act_cfg)
        self.norm3 = build_norm_layer(norm_cfg, out_features)[1]

        # 增加点卷积
        self.pw3x3 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=hidden_features,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.act4 = build_activation_layer(dw_act_cfg)
        self.norm4 = build_norm_layer(norm_cfg, hidden_features)[1]
        #

        # put the modules togather
        self.layers = [
            self.fc1, self.norm1, self.act1, self.dw3x3, self.norm2, self.act2,self.pw3x3,self.norm4,self.act4,
            self.fc2, self.norm3, self.act3
        ]

    def forward(self, x, H, W):
        """Forward function."""
        x = nlc_to_nchw(x, (H, W))
        for layer in self.layers:
            x = layer(x)
        x = nchw_to_nlc(x)
        return x