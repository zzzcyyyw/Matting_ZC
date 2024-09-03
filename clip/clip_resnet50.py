from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
import clip
from PIL import Image


""" ASPP模块 """
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, 1, 0, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # 参数冻结
        # self.freeze_layers()
        # embed_dim = width * 32  # the ResNet feature dimension
        # self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        self.batch_norm = True
        middle_chn = 2048

        # self.global_module = ASPP(middle_chn, atrous_rates=[6, 12, 18])
        # en_chn = middle_chn + 256   # aspp_channel

        self.deconv6_1 = nn.Conv2d(middle_chn, 512, kernel_size=1, bias=True)               # 512
        self.deconv5_1 = nn.Conv2d(1536, 512, kernel_size=5, padding=2, bias=True)      # 512
        self.deconv4_1 = nn.Conv2d(1024, 256, kernel_size=5, padding=2, bias=True)       # 512
        self.deconv3_1 = nn.Conv2d(512, 128, kernel_size=5, padding=2, bias=True)       # 256
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=True)

    # 参数冻结
    def freeze_layers(self):
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        conv_out = [x]
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        conv_out.append(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        # x = self.attnpool(x)

        conv5 = conv_out[-1]
        # global_ctx = self.global_module(conv5)
        # x = torch.cat([conv5, global_ctx], 1)
        x61d = F.relu(self.deconv6_1(conv5))

        x5d = F.interpolate(x61d, scale_factor=2, mode='bilinear', align_corners=False)
        x5d = torch.cat([x5d, conv_out[-2]], 1)
        x51d = F.relu(self.deconv5_1(x5d))

        x4d = F.interpolate(x51d, scale_factor=2, mode='bilinear', align_corners=False)
        x4d = torch.cat([x4d, conv_out[-3]], 1)
        x41d = F.relu(self.deconv4_1(x4d))

        x3d = F.interpolate(x41d, scale_factor=2, mode='bilinear', align_corners=False)
        x3d = torch.cat([x3d, conv_out[-4]], 1)
        x31d = F.relu(self.deconv3_1(x3d))

        x2d = F.interpolate(x31d, scale_factor=2, mode='bilinear', align_corners=False)
        # x2d = torch.cat([x2d, conv_out[-5]], 1)
        x21d = F.relu(self.deconv2_1(x2d))

        x1d = F.interpolate(x21d, scale_factor=2, mode='bilinear', align_corners=False)
        # x1d = torch.cat([x1d, conv_out[-6]], 1)
        x11d = F.relu(self.deconv1_1(x1d))

        raw_alpha = self.deconv1(x11d)
        pred_mattes = torch.sigmoid(raw_alpha)

        return pred_mattes


def model_builder(weight_init, clip_res50_weights):
    model = ModifiedResNet((3, 4, 6, 3), 1024, 32)
    model.apply(weight_init)
    model.load_state_dict(clip_res50_weights, strict=False)
    num_channels = 4
    if num_channels > 3:
        model_sd = model.state_dict()
        conv1_weights = model_sd['conv1.weight']

        c_out, c_in, h, w = conv1_weights.size()
        conv1_mod = torch.zeros(c_out, num_channels, h, w)
        torch.nn.init.xavier_normal_(conv1_mod)
        conv1_mod[:, :3, :, :] = conv1_weights
        conv1 = model.conv1
        conv1.in_channels = num_channels
        conv1.weight = torch.nn.Parameter(conv1_mod)

        model.conv1 = conv1
        model_sd['conv1.weight'] = conv1_mod
        model.load_state_dict(model_sd)
    return model


if __name__ == "__main__":
    model = ModifiedResNet((3, 4, 6, 3), 1024, 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_weights, preprocess = clip.load('RN50', device=device)

    weights_state = clip_weights.visual.state_dict()

    new_weights_state = weights_state.copy()
    for key in list(new_weights_state.keys()):
        if 'attnpool' in key:
            del new_weights_state[key]


    model.load_state_dict(new_weights_state, strict=False)

    model.to(device)
    image = preprocess(Image.open("0_trimap.png")).unsqueeze(0).to(device)
    input = F.interpolate(image, size=(320, 320), mode='bilinear', align_corners=False)
    out = model(input)
    out = out.view(out.size(0), -1, 32, 32)

    print(out)
