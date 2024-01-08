import argparse
from models.unet_parts import *
from models.DS_parts import *
from models.layers import *
from models.RR_parts import *
import pytorch_lightning as pl
from models.regression_lightning import Precip_regression_base
from cloud_cover.cloud_cover_lightning import Cloud_base


class UNet_precip(Precip_regression_base):
    def __init__(self, hparams):
        super(UNet_precip, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.inc = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class SmaAt_UNet_cloud(Cloud_base):
    def __init__(self, hparams,  kernels_per_layer=1, reduction_ratio = 16):
        super(SmaAt_UNet_cloud, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.inc = DoubleConvDS(self.in_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


class CAtUnet(Cloud_base): 
    def __init__(self, hparams, kernels_per_layer=1, reduction_ratio=16):
        super(CAtUnet, self).__init__(hparams=hparams)
        self.in_channels = hparams['in_channels']
        self.out_channels = hparams['out_channels']
        self.bilinear = hparams['bilinear']

        self.inc = DoubleConv(self.in_channels, 48)
        self.CSPA1 = CSPA(48)
        self.lstm1 = RNN([4, 4], 4, 256)
        # self.lstmConv1 = nn.Conv2d(4, 48, kernel_size=3, padding=1)
        self.down1 = Down(48, 96)
        self.CSPA2 = CSPA(96)
        self.lstm2 = RNN([8, 8], 8, 128)
        # self.lstmConv2 = nn.Conv2d(8, 96, kernel_size=3, padding=1)
        self.down2 = Down(96, 192)
        self.CSPA3 = CSPA(192)
        self.lstm3 = RNN([16, 16], 16, 64)
        # self.lstmConv3 = nn.Conv2d(16, 192, kernel_size=3, padding=1)
        self.down3 = Down(192, 384)
        self.CSPA4 = CSPA(384)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(384, 768 // factor)
        self.up1 = Up(768, 384 // factor, self.bilinear)
        self.up2 = LSTMUp(576, 192 // factor, self.bilinear)
        self.up3 = LSTMUp(288, 96 // factor, self.bilinear)
        self.up4 = LSTMUp(144, 48, self.bilinear)

        self.outc = OutConv(48, self.out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.inc(x)
        x1Attcspa = self.CSPA1(x1)
        x2 = self.down1(x1)
        x2Attcspa = self.CSPA2(x2)
        x3 = self.down2(x2)
        x3Attcspa = self.CSPA3(x3)
        x4 = self.down3(x3)
        x4Attcspa = self.CSPA4(x4)
        x5 = self.down4(x4)

        tx1 = x1.reshape(b, 12, 4, w, h)
        tx2 = x2.reshape(b, 12, 8, int(w/2), int(h/2))
        tx3 = x3.reshape(b, 12, 16, int(w/4), int(h/4))
        # print(tx1.shape)
        lstm1 = self.lstm1(tx1)
        lstm2 = self.lstm2(tx2)
        lstm3 = self.lstm3(tx3)
        # print(len(lstm1))
        lstm1 = lstm1.reshape(b, 48, w, h)
        lstm2 = lstm2.reshape(b, 96, int(w/2), int(h/2))
        lstm3 = lstm3.reshape(b, 192, int(w/4), int(h/4))

        x = self.up1(x5, x4Attcspa)
        x = self.up2(x, x3Attcspa, lstm3)
        x = self.up3(x, x2Attcspa, lstm2)
        x = self.up4(x, x1Attcspa, lstm1)
        logits = self.outc(x)
        return logits




