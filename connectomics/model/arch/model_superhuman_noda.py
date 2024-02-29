'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2023-02-17 10:58:28
LastEditTime: 2023-04-06 22:02:34
'''
# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import conv3dBlock, upsampleBlock
from .residual import resBlock_pni, resBlock_pni_iso

def mean_std(x, eps=1e-6):
    mu = x.mean(dim=[2, 3, 4], keepdim=True)
    var = x.var(dim=[2, 3, 4], keepdim=True)
    sig = (var + eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    return mu, sig

class UNet_PNI_aniso(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                    out_planes=1,
                    filters=[28, 36, 48, 64, 80],    # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                    upsample_mode='transposeS',    # transposeS, bilinear
                    decode_ratio=1,
                    merge_mode='cat',    # cat, add
                    pad_mode='zero',
                    bn_mode='async',    # async or sync
                    relu_mode='elu',
                    init_mode='kaiming_normal',
                    bn_momentum=0.001,
                    if_sigmoid=True,
                    show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_aniso, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.output = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.output(embed_out)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            return out, embed_out
        else:
            return out


class UNet_PNI_iso(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                    out_planes=1,
                    filters=[28, 36, 48, 64, 80],    # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                    upsample_mode='transposeS',    # transposeS, bilinear
                    decode_ratio=1,
                    merge_mode='cat',    # cat, add
                    pad_mode='zero',
                    bn_mode='async',    # async or sync
                    relu_mode='elu',
                    init_mode='kaiming_normal',
                    bn_momentum=0.001,
                    if_sigmoid=True,
                    show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_iso, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(5, 5, 5)], 
                                    [1], 
                                    [(2, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # self.instance_norm = nn.InstanceNorm3d(filters2[0])

        # downsample stream
        self.conv0 = resBlock_pni_iso(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        self.conv1 = resBlock_pni_iso(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        self.conv2 = resBlock_pni_iso(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        self.conv3 = resBlock_pni_iso(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((2, 2, 2), (2, 2, 2))

        self.center = resBlock_pni_iso(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (2,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni_iso(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni_iso(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (2,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni_iso(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni_iso(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (2,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni_iso(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni_iso(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (2,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni_iso(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni_iso(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out_pre = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(5, 5, 5)], 
                                        [1], 
                                        [(2, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.embed_out_post = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(5, 5, 5)], 
                                        [1], 
                                        [(2, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.output_pre = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.output_post = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)

        self.sigmoid = nn.Sigmoid()

    def forward_main(self, conv0, dp=0.5):
        pool0 = self.pool0(conv0)      # [28, 80, 80, 80]
        pool0 = F.dropout3d(pool0, p=dp)
        conv1 = self.conv1(pool0)      # [36, 80, 80, 80]
        pool1 = self.pool1(conv1)      # [36, 40, 40, 40]
        pool1 = F.dropout3d(pool1, p=dp)
        conv2 = self.conv2(pool1)      # [48, 40, 40, 40]
        pool2 = self.pool2(conv2)      # [48, 20, 20, 20]
        pool2 = F.dropout3d(pool2, p=dp)
        conv3 = self.conv3(pool2)      # [64, 20, 20, 20]
        pool3 = self.pool3(conv3)      # [64, 10, 10, 10]
        pool3 = F.dropout3d(pool3, p=dp)

        center = self.center(pool3)    # [80, 10, 10, 10]

        up0 = self.up0(center)               # [64, 20, 20, 20]
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)    # [64, 20, 20, 20]
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)             # [64, 20, 20, 20]

        up1 = self.up1(conv4)                # [48, 40, 40, 40]
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)    # [48, 40, 40, 40]
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)             # [48, 40, 40, 40]

        up2 = self.up2(conv5)                # [36, 80, 80, 80]
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)    # [36, 80, 80, 80]
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)             # [36, 80, 80, 80]

        up3 = self.up3(conv6)                # [28, 160, 160, 160]
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)    # [28, 160, 160, 160]
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)             # [28, 160, 160, 160]

        embed_out_pre = self.embed_out_pre(conv7)    # [28, 160, 160, 160]
        out_pre = self.output_pre(embed_out_pre)         # [1, 160, 160, 160]

        embed_out_post = self.embed_out_post(conv7)    # [28, 160, 160, 160]
        out_post = self.output_pre(embed_out_post)         # [1, 160, 160, 160]

        if self.if_sigmoid:
            out_pre = torch.sigmoid(out_pre)
            out_post = torch.sigmoid(out_post)

        if self.show_feature:
            return out_pre, out_post, conv7
        else:
            return out_pre, out_post

    def forward(self, x):
        x_src = x[:, 0:1, ...]
        if self.training:
            x_src = x[:, 0:1, ...]
            embed = self.embed_in(x_src)
            conv0 = self.conv0(embed)
            out_pre, out_post = self.forward_main(conv0)
            return out_pre, out_post
        else:
            embed = self.embed_in(x)
            conv0 = self.conv0(embed)
            out_pre, out_post = self.forward_main(conv0)
            return out_pre, out_post


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import numpy as np
    from ptflops import get_model_complexity_info

    # input = np.random.random((1,1,18,160,160)).astype(np.float32)
    input = np.random.random((1,1,160,160,160)).astype(np.float32)
    x = torch.tensor(input)#.cuda()

    # model = UNet_PNI_aniso(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='add')    # 87.27 GMac, 1.48 M
    # model = UNet_PNI_aniso(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='cat')    # 92.78 GMac, 1.55 M
    # model = UNet_PNI_aniso(filters=[28, 36, 48, 64, 80], upsample_mode='transposeS', merge_mode='add')    # 87.37 GMac, 1.48 M
    # model = UNet_PNI_aniso(filters=[32, 64, 128, 256, 512], upsample_mode='bilinear', merge_mode='add')    # 275.76 GMac, 26.13 M

    model = UNet_PNI_iso(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='add')#.cuda()    # 1077.1 GMac, 1.92 M

    # macs, params = get_model_complexity_info(model, (1, 160, 160, 160), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    out = model(x)
    print(out.shape)