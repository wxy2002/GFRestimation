import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样层，最大池化然后双卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样层，转置卷积或上采样+双卷积"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值进行上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入可能不是2的整数倍，需要进行裁剪
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 连接通道
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SEModule(nn.Module):
    """Squeeze-and-Excitation模块，用于通道注意力"""
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        # 确保reduction后的通道数至少为1
        reduction_channel = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Squeeze操作
        y = self.avg_pool(x).view(b, c)
        # Excitation操作
        y = self.fc(y).view(b, c, 1, 1, 1)
        # 重新加权
        return x * y.expand_as(x)

class AdaptivePoolingModule(nn.Module):
    """自适应池化 + 1x1卷积模块，用于特征提取"""
    def __init__(self, in_channels, out_channels=16, pool_size=4):
        super(AdaptivePoolingModule, self).__init__()
        
        self.se = SEModule(in_channels)

        self.features = nn.Sequential(
            # 使用1x1卷积调整通道数
            nn.Conv3d(in_channels, in_channels // 2 if in_channels > 1 else 1, kernel_size=1, bias=False),
            # nn.BatchNorm2d(in_channels // 2 if in_channels > 1 else 1),
            nn.ReLU(),
            # 使用自适应平均池化将特征图调整为固定大小
            nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size)),
            # 再次使用1x1卷积进一步降维
            nn.Conv3d(in_channels // 2 if in_channels > 1 else 1, out_channels, kernel_size=1),
            nn.ReLU(),
            # 展平为一维向量
            nn.Flatten(),
            nn.Linear(1024, 16),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.se(x)
        return self.features(x)

class UNet3D(nn.Module):
    """
    简化版UNet模型，直接处理2D数据
    输入形状: (batch_size, 64, 256, 256)
    """
    def __init__(self, n_channels=64, n_classes=1, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # UNet编码器部分
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        
        # UNet解码器部分
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        # 用于特征提取的自适应池化 + 1x1卷积模块
        self.f1 = AdaptivePoolingModule(in_channels=64, out_channels=16, pool_size=4)
        self.f2 = AdaptivePoolingModule(in_channels=32, out_channels=16, pool_size=4)
        self.f3 = AdaptivePoolingModule(in_channels=16, out_channels=16, pool_size=4)
        self.f4 = AdaptivePoolingModule(in_channels=16, out_channels=16, pool_size=4)
        self.f0 = AdaptivePoolingModule(in_channels=1, out_channels=16, pool_size=4)
        
        # 最终预测层
        self.final_layer_l = nn.Sequential(
            nn.Linear(16 * 5, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

        self.final_layer_r = nn.Sequential(
            nn.Linear(16 * 5, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x, clin):
        """
        x: 输入形状 (batch_size, 64, 256, 256)
        clin: 临床特征 (batch_size, 5)
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, 1, 32, 256, 256)  # 将64个切片作为通道处理
        
        # 将64个切片作为通道处理
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        f1_out = self.f1(x)
        x = self.up2(x, x3)
        f2_out = self.f2(x)
        x = self.up3(x, x2)
        f3_out = self.f3(x)
        x = self.up4(x, x1)
        f4_out = self.f4(x)
        x = self.outc(x)
        flat_features = self.f0(x)
        flat_features = torch.cat([f1_out, f2_out, f3_out, f4_out, flat_features], dim=1)

        # 最终预测
        output_l = self.final_layer_l(flat_features)
        output_r = self.final_layer_r(flat_features)

        output = torch.cat([output_l, output_r], dim=1)  # 合并左右肾的输出
        return output