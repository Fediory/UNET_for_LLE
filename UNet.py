import torch.nn as nn

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


# 定义 U-Net 网络
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器部分
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
           
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器部分
        self.decoder4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        
        encoder_output = self.encoder1(x)
        encoder_output2 = self.encoder2(encoder_output)
        encoder_output3 = self.encoder3(encoder_output2)
        encoder_output4 = self.encoder4(encoder_output3)
        
        decoder_output4 = self.decoder4(encoder_output4) + encoder_output3
        decoder_output3 = self.decoder3(decoder_output4) + encoder_output2
        decoder_output2 = self.decoder2(decoder_output3) + encoder_output
        decoder_output = self.decoder1(decoder_output2) + x
        
        return decoder_output
