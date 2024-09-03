import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from config import device


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
            # nn.BatchNorm2d(out_channels),
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
            res.append(conv(x.type(torch.float32)))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ImageEncoder(nn.Module):
    def __init__(self, ori_encoder):
        super(ImageEncoder, self).__init__()

        # 加载预训练resnet encoder
        self.conv1 = ori_encoder.visual.conv1
        self.bn1 = ori_encoder.visual.bn1
        self.relu1 = ori_encoder.visual.relu1

        self.conv2 = ori_encoder.visual.conv2
        self.bn2 = ori_encoder.visual.bn2
        self.relu2 = ori_encoder.visual.relu2

        self.conv3 = ori_encoder.visual.conv3
        self.bn3 = ori_encoder.visual.bn3
        self.relu3 = ori_encoder.visual.relu3

        # 去掉了attnpool层
        self.avgpool = ori_encoder.visual.avgpool
        self.layer1 = ori_encoder.visual.layer1
        self.layer2 = ori_encoder.visual.layer2
        self.layer3 = ori_encoder.visual.layer3
        self.layer4 = ori_encoder.visual.layer4

    def forward(self, x_rgb):
        conv_out = [x_rgb]

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x_rgb.type(self.conv1.weight.dtype)
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

        return conv_out

# 替代clip中的text encoder
class DepthEncoder(nn.Module):
    def __init__(self, ori_encoder, embed_dim):
        super(DepthEncoder, self).__init__()

        # 将深度图变形使其满足text encoder的输入格式
        self.embed_dim = embed_dim
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=1)
        self.adaptiva_pool = nn.AdaptiveAvgPool2d((7, 11))
        self.depth_transform_conv = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)

        # 加载预训练text_encoder
        self.transformer = ori_encoder.transformer.resblocks
        self.ln_final = ori_encoder.ln_final

    def forward(self, depth_map, conv5):
        x_d = self.adaptiva_pool(depth_map)
        x_d = self.conv1x1(x_d)
        x_d = x_d.permute(0, 2, 3, 1).reshape(x_d.size(0), -1, self.embed_dim)
        x_d = x_d.permute(1, 0, 2).to(torch.float16)
        x_d = self.transformer(x_d)
        x_d = x_d.permute(1, 0, 2)
        x_d = self.ln_final(x_d)
        x_d = x_d.permute(0, 2, 1).reshape(x_d.size(0), -1, 7, 11)
        x_d = F.interpolate(x_d, size=(conv5.size(2), conv5.size(3)), mode='bilinear', align_corners=False)
        return x_d



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.batch_norm = True
        middle_chn = 2048

        self.global_module = ASPP(middle_chn, atrous_rates=[6, 12, 18])
        en_chn = middle_chn + 256 + 512     # aspp_channel + depth_features_channel

        self.deconv6_1 = nn.Conv2d(en_chn, 512, kernel_size=1, bias=True)  # 512
        self.deconv5_1 = nn.Conv2d(1536, 512, kernel_size=5, padding=2, bias=True)  # 512
        self.deconv4_1 = nn.Conv2d(1024, 256, kernel_size=5, padding=2, bias=True)  # 512
        self.deconv3_1 = nn.Conv2d(512, 128, kernel_size=5, padding=2, bias=True)  # 256
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=True)

    def forward(self, conv_out, depth_features):
        conv5 = conv_out[-1]
        global_ctx = self.global_module(conv5)
        x = torch.cat([conv5, global_ctx, depth_features], 1)
        x61d = F.relu(self.deconv6_1(x))

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


class ClipMattingModule(nn.Module):
    def __init__(self, img_enc, depth_enc, net_dec):
        super(ClipMattingModule, self).__init__()
        # self.ori_encoder = ori_encoder

        self.img_encoder = img_enc
        self.depth_encoder = depth_enc
        self.decoder = net_dec

        # 冻结clip中 image encoder 和 text encoder 的预训练权重
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        for param in self.depth_encoder.transformer.parameters():
            param.requires_grad = False
        for param in self.depth_encoder.ln_final.parameters():
            param.requires_grad = False

    @property
    def dtype(self):
        return self.ori_encoder.visual.conv1.weight.dtype

    def forward(self, img, depth_map):
        conv_out = self.img_encoder(img)
        depth_features = self.depth_encoder(depth_map, conv_out[-1])
        alpha_out = self.decoder(conv_out, depth_features)

        return alpha_out


def model_builder(weight_init):
    ori_encoder, _ = clip.load('RN50', device=device)
    image_encoder = ImageEncoder(ori_encoder)
    depth_encoder = DepthEncoder(ori_encoder, embed_dim=512)
    net_decoder = Decoder()
    net_decoder.apply(weight_init)      # 将解码器进行初始化
    model = ClipMattingModule(image_encoder, depth_encoder, net_decoder)

    return model


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = torch.randn(32, 3, 320, 320).to(device)
    depth = torch.randn(32, 1, 320, 320).to(device)

    model = model_builder(weight_init)
    model.to(device)
    model.train()

    out = model(image, depth)
    out = out.view(out.size(0), -1, 32, 32)

    print(out)
