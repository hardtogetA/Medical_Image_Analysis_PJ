import torch
from torch import nn
import torch.nn.functional as F
import copy
import model.resnet as models
from model.module.decoder import build_decoder
from model.module.ASPP import ASPP
from torchvision.models import resnet18

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        layers = args.layers
        classes = args.classes
        sync_bn = args.sync_bn
        pretrained = True
        assert layers in [50, 101, 152]
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        self.pyramid = args.pyramid

        models.BatchNorm = BatchNorm

        print('INFO: Using ResNet {}'.format(layers))
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3),
        )

        reduce_dim = 256
        fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.Dropout2d(p=0.5)
        )

        # Using Feature Enrichment Module from PFENet as context module
        if self.pyramid:

            self.pyramid_bins = args.ppm_scales
            self.avgpool_list = []

            for bin in self.pyramid_bins:
                if bin > 1:
                    self.avgpool_list.append(
                        nn.AdaptiveAvgPool2d(bin)
                    )

            self.corr_conv = []
            self.beta_conv = []
            self.inner_cls = []

            for bin in self.pyramid_bins:
                self.beta_conv.append(nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True)
                ))
                self.inner_cls.append(nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.1),
                    nn.Conv2d(reduce_dim, classes, kernel_size=1)
                ))
            self.beta_conv = nn.ModuleList(self.beta_conv)
            self.inner_cls = nn.ModuleList(self.inner_cls)

            self.alpha_conv = []
            for idx in range(len(self.pyramid_bins) - 1):
                self.alpha_conv.append(nn.Sequential(
                    nn.Conv2d(2 * reduce_dim, reduce_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.ReLU(inplace=True)
                ))
            self.alpha_conv = nn.ModuleList(self.alpha_conv)

            self.res1 = nn.Sequential(
                nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.res2 = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

        # Using ASPP as context module
        else:
            self.ASPP = ASPP(out_channels=reduce_dim)
            self.skip1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.skip2 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.skip3 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.decoder = build_decoder(256)
            self.cls_aux = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout2d(p=0.1),
                                         nn.Conv2d(reduce_dim, classes, kernel_size=1))

    def forward(self, x, y=None, cls=None):
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        # x_ = copy.deepcopy(x)
        out_i = torch.zeros_like(x).to(x.device)
        with torch.no_grad():
            x_0 = self.layer0(x)
            x_1 = self.layer1(x_0)
            x_2 = self.layer2(x_1)
            x_3 = self.layer3(x_2)

            # x = torch.cat([x_3, x_2], 1)
            # x = self.down_conv(x)

            # if self.pyramid:
            #     out_list = []
            #     pyramid_feat_list = []

            #     for idx, tmp_bin in enumerate(self.pyramid_bins):
            #         if tmp_bin <= 1.0:
            #             bin = int(x.shape[2] * tmp_bin)
            #             x_tmp = nn.AdaptiveAvgPool2d(bin)(x)
            #         else:
            #             bin = tmp_bin
            #             x_tmp = self.avgpool_list[idx](x)

            #         if idx >= 1:
            #             pre_feat_bin = pyramid_feat_list[idx - 1].clone()
            #             pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
            #             rec_feat_bin = torch.cat([x_tmp, pre_feat_bin], 1)
            #             x_tmp = self.alpha_conv[idx - 1](rec_feat_bin) + x_tmp

            #         x_tmp = self.beta_conv[idx](x_tmp) + x_tmp
            #         inner_out_bin = self.inner_cls[idx](x_tmp)
            #         x_tmp = F.interpolate(x_tmp, size=(x.size(2), x.size(3)),
            #                                     mode='bilinear', align_corners=True)
            #         pyramid_feat_list.append(x_tmp)
            #         out_list.append(inner_out_bin)

            #     final_feat = torch.cat(pyramid_feat_list, 1)
            #     final_feat = self.res1(final_feat)
            #     final_feat = self.res2(final_feat) + final_feat
            #     out = self.cls(final_feat)

            # # ASPP structure
            # else:
            #     final_feat = x + self.skip1(x)
            #     final_feat = final_feat + self.skip2(final_feat)
            #     final_feat = final_feat + self.skip3(final_feat)
            #     final_feat = self.ASPP(final_feat)
            #     decoder_out = self.decoder(final_feat, x_1)
            #     out = self.cls(decoder_out)

            # if self.zoom_factor != 1:
            #     out_i = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        # classification
        # x_ = self.layer4(x_3 * F.interpolate(out.argmax(1,keepdim=True).float(), size=x_3.shape[-2:], mode='bilinear', align_corners=True))
        x_ = self.layer4(x_3)
        predict_cls = self.cls_head(x_)

        if self.training:
            main_loss = nn.functional.cross_entropy(predict_cls, cls)
            aux_loss = torch.zeros_like(main_loss).cuda()

            # if self.pyramid:
            #     for idx_k in range(len(out_list)):
            #         inner_out = out_list[idx_k]
            #         inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
            #         aux_loss = aux_loss + self.criterion(inner_out, y.long())
            #     aux_loss = aux_loss / len(out_list)
            # else:
            #     aux_out = self.cls_aux(final_feat)
            #     aux_out = F.interpolate(aux_out, size=(h, w), mode='bilinear', align_corners=True)
            #     aux_loss = self.criterion(aux_out, y.long())

            return out_i.max(1)[1], predict_cls, main_loss, aux_loss
        else:
            return out_i, predict_cls

    def _optimizer(self, args):
        optimizer = torch.optim.Adam(
                [
                    # {'params': self.layer0.parameters()},
                    # {'params': self.layer1.parameters()},
                    # {'params': self.layer2.parameters()},
                    # {'params': self.layer3.parameters()},
                    {'params': self.layer4.parameters()},
                    {'params': self.cls_head.parameters()},
                ],
                lr=args.base_lr, weight_decay=args.weight_decay)
        return optimizer
