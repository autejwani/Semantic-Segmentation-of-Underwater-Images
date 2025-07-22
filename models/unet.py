import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, in_ch, gating_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=8):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = ConvBlock(512, 1024)

        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att6 = AttentionBlock(in_ch=512, gating_ch=512, inter_ch=256)
        self.dec6 = ConvBlock(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att7 = AttentionBlock(in_ch=256, gating_ch=256, inter_ch=128)
        self.dec7 = ConvBlock(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att8 = AttentionBlock(in_ch=128, gating_ch=128, inter_ch=64)
        self.dec8 = ConvBlock(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att9 = AttentionBlock(in_ch=64, gating_ch=64, inter_ch=32)
        self.dec9 = ConvBlock(128, 64)

        # Final output
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        e5 = self.enc5(p4)

        # Decoder
        d6 = self.up6(e5)
        e4_att = self.att6(e4, d6)
        d6 = torch.cat((e4_att, d6), dim=1)
        d6 = self.dec6(d6)

        d7 = self.up7(d6)
        e3_att = self.att7(e3, d7)
        d7 = torch.cat((e3_att, d7), dim=1)
        d7 = self.dec7(d7)

        d8 = self.up8(d7)
        e2_att = self.att8(e2, d8)
        d8 = torch.cat((e2_att, d8), dim=1)
        d8 = self.dec8(d8)

        d9 = self.up9(d8)
        e1_att = self.att9(e1, d9)
        d9 = torch.cat((e1_att, d9), dim=1)
        d9 = self.dec9(d9)

        out = self.final_conv(d9)
        return out
