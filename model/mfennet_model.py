### MFEnNet
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel_size=3, act_type="mish"):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act_type = act_type.lower()
        if self.act_type == "leakyrelu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif self.act_type == "tanh":
            self.act = nn.Tanh()
        elif self.act_type == "mish":
            # Mish: x * tanh(softplus(x))
            self.act = None  # sẽ xử lý trong forward
        elif self.act_type == "swish":
            # Swish: x * sigmoid(x)
            self.act = None  # sẽ xử lý trong forward
        else:
            self.act = nn.ReLU(inplace=True)  # fallback

    def forward(self, x):
        x = self.conv(x)
        if self.act_type == "mish":
            # Mish: x * tanh(softplus(x))
            return x * torch.tanh(F.softplus(x))
        elif self.act_type == "swish":
            # Swish: x * sigmoid(x)
            return x * torch.sigmoid(x)
        elif self.act is not None:
            return self.act(x)
        else:
            return x


class PoolingTransformerBlock(nn.Module):
    def __init__(self, emb_dim, mlp_ratio):
        super(PoolingTransformerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, emb_dim)  # Use GroupNorm instead of LayerNorm for 4D tensors
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 3x3 pooling
        self.norm2 = nn.GroupNorm(1, emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(emb_dim * mlp_ratio, emb_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Normalize before token mixing
        x_norm = self.norm1(x)
        # Token mixing with 3x3 average pooling
        mixed = self.pool(x_norm)
        # Add residual connection
        x = x + mixed

        # Normalize before feed-forward network
        x_norm = self.norm2(x)
        
        # Reshape for linear layers: (B, C, H, W) -> (B*H*W, C)
        x_norm = x_norm.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Feed-forward network
        ffn_out = self.ffn(x_norm)
        
        # Reshape back: (B*H*W, C) -> (B, C, H, W)
        ffn_out = ffn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Add residual connection
        out = x + ffn_out

        return out



class MFEnNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_filters_1st=32):
        super(MFEnNet, self).__init__()

        self.conv1_1 = ConvBlock(n_channels, num_filters_1st, 3, act_type="swish")
        self.pte1_1 = PoolingTransformerBlock(num_filters_1st, 4)
        self.pte1_2 = PoolingTransformerBlock(num_filters_1st, 4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBlock(num_filters_1st, int(2*num_filters_1st), 3, act_type="swish")
        self.pte2_1 = PoolingTransformerBlock(int(2*num_filters_1st), 4)
        self.pte2_2 = PoolingTransformerBlock(int(2*num_filters_1st), 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvBlock(int(2*num_filters_1st), int(4*num_filters_1st), 3, act_type="swish")
        self.pte3_1 = PoolingTransformerBlock(int(4*num_filters_1st), 4)
        self.pte3_2 = PoolingTransformerBlock(int(4*num_filters_1st), 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = ConvBlock(int(4*num_filters_1st), int(8*num_filters_1st), 3, act_type="swish")
        self.pte4_1 = PoolingTransformerBlock(int(8*num_filters_1st), 4)
        self.pte4_2 = PoolingTransformerBlock(int(8*num_filters_1st), 4)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = ConvBlock(int(8*num_filters_1st), int(16*num_filters_1st), 3, act_type="swish")
        self.pte5_1 = PoolingTransformerBlock(int(16*num_filters_1st), 4)
        self.pte5_2 = PoolingTransformerBlock(int(16*num_filters_1st), 4)
        self.spp_conv = ConvBlock(int(16*num_filters_1st), int(8*num_filters_1st), 1, act_type="swish")

        # SPP pooling layers
        self.spp_pool_13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.spp_pool_9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.spp_pool_5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.spp_concat_conv = ConvBlock(int(8*num_filters_1st)*4, int(16*num_filters_1st), 1, act_type="swish")
        self.drop5 = nn.Dropout(0.5)

        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up6_conv = ConvBlock(int(16*num_filters_1st), int(8*num_filters_1st), 1, act_type="swish")
        self.conv6_1 = ConvBlock(int(8*num_filters_1st)*2, int(8*num_filters_1st), 3, act_type="swish")
        self.conv6_2 = ConvBlock(int(8*num_filters_1st), int(8*num_filters_1st), 3, act_type="swish")
        self.conv6_3 = ConvBlock(int(8*num_filters_1st), int(8*num_filters_1st), 3, act_type="swish")

        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up7_conv = ConvBlock(int(8*num_filters_1st), int(4*num_filters_1st), 1, act_type="swish")
        self.conv7_1 = ConvBlock(int(4*num_filters_1st)*2, int(4*num_filters_1st), 3, act_type="swish")
        self.conv7_2 = ConvBlock(int(4*num_filters_1st), int(4*num_filters_1st), 3, act_type="swish")
        self.conv7_3 = ConvBlock(int(4*num_filters_1st), int(4*num_filters_1st), 3, act_type="swish")

        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8_conv = ConvBlock(int(4*num_filters_1st), int(2*num_filters_1st), 1, act_type="swish")
        self.conv8_1 = ConvBlock(int(2*num_filters_1st)*2, int(2*num_filters_1st), 3, act_type="swish")
        self.conv8_2 = ConvBlock(int(2*num_filters_1st), int(2*num_filters_1st), 3, act_type="swish")
        self.conv8_3 = ConvBlock(int(2*num_filters_1st), int(2*num_filters_1st), 3, act_type="swish")

        self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up9_conv = ConvBlock(int(2*num_filters_1st), num_filters_1st, 1, act_type="swish")
        self.conv9_1 = ConvBlock(num_filters_1st*2, num_filters_1st, 3, act_type="swish")
        self.conv9_2 = ConvBlock(num_filters_1st, num_filters_1st, 3, act_type="swish")
        self.conv9_3 = ConvBlock(num_filters_1st, num_filters_1st, 3, act_type="swish")

        self.outc = OutConv(num_filters_1st, n_classes)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1_1(x)
        conv1 = self.pte1_1(conv1)
        conv1 = self.pte1_2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.pte2_1(conv2)
        conv2 = self.pte2_2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.pte3_1(conv3)
        conv3 = self.pte3_2(conv3)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = self.pte4_1(conv4)
        conv4 = self.pte4_2(conv4)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5_1(pool4)
        conv5 = self.pte5_1(conv5)
        conv5 = self.pte5_2(conv5)
        conv5 = self.spp_conv(conv5)

        # SPP
        spp1 = conv5
        spp2 = self.spp_pool_13(conv5)
        spp3 = self.spp_pool_9(conv5)
        spp4 = self.spp_pool_5(conv5)
        spp_cat = torch.cat([spp1, spp2, spp3, spp4], dim=1)
        conv5 = self.spp_concat_conv(spp_cat)
        drop5 = self.drop5(conv5)

        # Decoder
        up6 = self.up6(drop5)
        up6 = self.up6_conv(up6)
        merge6 = torch.cat([drop4, up6], dim=1)
        conv6 = self.conv6_1(merge6)
        conv6 = self.conv6_2(conv6)
        conv6 = self.conv6_3(conv6)

        up7 = self.up7(conv6)
        up7 = self.up7_conv(up7)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7_1(merge7)
        conv7 = self.conv7_2(conv7)
        conv7 = self.conv7_3(conv7)

        up8 = self.up8(conv7)
        up8 = self.up8_conv(up8)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8_1(merge8)
        conv8 = self.conv8_2(conv8)
        conv8 = self.conv8_3(conv8)

        up9 = self.up9(conv8)
        up9 = self.up9_conv(up9)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9_1(merge9)
        conv9 = self.conv9_2(conv9)
        conv9 = self.conv9_3(conv9)

        logits = self.outc(conv9)

        return logits