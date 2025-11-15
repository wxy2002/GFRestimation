import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
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
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1插值到x2的大小
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=False)
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
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        # 确保reduction后的通道数至少为1
        reduction_channel = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, reduction_channel, 1),
            nn.ReLU(),
            nn.Conv3d(reduction_channel, channel, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        max_result = self.max_pool(x)
        avg_result = self.avg_pool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid((max_out + avg_out) / 2)
        return self.relu(x * output + x)

class final_mlp(nn.Module):
    def __init__(self, input_dim):
        super(final_mlp, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PooltoLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PooltoLinear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

class UNet3D(nn.Module):
    """
    简化版UNet模型，直接处理3D数据
    输入形状: (batch_size, 64, 256, 256)
    """
    def __init__(self, n_channels=64, n_classes=1, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.out_channels = 4
        self.out_shape = 16
        self.pool_size = 2

        # UNet编码器部分
        self.inc = DoubleConv(n_channels, 16)
        self.semd1 = SEModule(16)
        self.down1 = Down(16, 32)
        self.semd2 = SEModule(32)
        self.down2 = Down(32, 64)
        self.semd3 = SEModule(64)
        self.down3 = Down(64, 128)
        self.semd4 = SEModule(128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.semd5 = SEModule(256 // factor)
        
        # UNet解码器部分
        self.up1 = Up(256, 128 // factor, bilinear)
        self.semu1 = SEModule(128 // factor)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.semu2 = SEModule(64 // factor)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.semu3 = SEModule(32 // factor)
        self.up4 = Up(32, 16, bilinear)
        self.semu4 = SEModule(16)
        self.outc = OutConv(16, n_classes)

        self.sem_feature = nn.ModuleList([
            SEModule(16),
            SEModule(32),
            SEModule(64),
            SEModule(128),
            SEModule(256 // factor),
            SEModule(128 // factor),
            SEModule(64 // factor),
            SEModule(32 // factor),
            SEModule(16)
        ])
        self.pool_to_linear = nn.ModuleList([
            PooltoLinear(16, self.out_channels),
            PooltoLinear(32, self.out_channels),
            PooltoLinear(64, self.out_channels),
            PooltoLinear(128, self.out_channels),
            PooltoLinear(256 // factor, self.out_channels),
            PooltoLinear(128 // factor, self.out_channels),
            PooltoLinear(64 // factor, self.out_channels),
            PooltoLinear(32 // factor, self.out_channels),
            PooltoLinear(16, self.out_channels)
        ])

        # 最终预测层
        self.final_layer_l = final_mlp(input_dim=self.out_channels * 9)
        self.final_layer_r = final_mlp(input_dim=self.out_channels * 9)
    
    def extract_multiscale_features(self, feature_maps):
        """多尺度特征提取和融合"""
        features = []
        for i, fm in enumerate(feature_maps):
            # 使用多种池化方式
            # emf = self.sem_feature[i](fm)
            avg_pool = F.adaptive_avg_pool3d(fm, 1).flatten(1)
            max_pool = F.adaptive_max_pool3d(fm, 1).flatten(1)
            pool_concat = torch.cat([avg_pool, max_pool], dim=1)
            features.append(self.pool_to_linear[i](pool_concat))
        return torch.cat(features, dim=1)

    def forward(self, x, clin=None):
        """
        x: 输入形状 (batch_size, D, H, W)
        clin: 临床特征 (batch_size, 2)
        """
        batch_size = x.size(0)
        x = x.reshape(batch_size, 1, 16, 128, 128) # 为了进行3D卷积，reshape一个C通道
        
        # 将64个切片作为通道处理
        x1 = self.inc(x)
        x1 = self.semd1(x1)
        x2 = self.down1(x1)
        x2 = self.semd2(x2)
        x3 = self.down2(x2)
        x3 = self.semd3(x3)
        x4 = self.down3(x3)
        x4 = self.semd4(x4)
        x5 = self.down4(x4)
        x5 = self.semd5(x5)
        
        features_maps = [x1, x2, x3, x4, x5]
        # features_maps = []
        x = self.up1(x5, x4)
        x = self.semu1(x)
        features_maps.append(x)
        x = self.up2(x, x3)
        x = self.semu2(x)
        features_maps.append(x)
        x = self.up3(x, x2)
        x = self.semu3(x)
        features_maps.append(x)
        x = self.up4(x, x1)
        x = self.semu4(x)
        features_maps.append(x)
        x = self.outc(x)

        features = self.extract_multiscale_features(feature_maps=features_maps)

        # 最终预测
        output_l = self.final_layer_l(features)
        output_r = self.final_layer_r(features)
        output = torch.cat([output_l, output_r], dim=1)  # 合并左右肾的输出
        return output