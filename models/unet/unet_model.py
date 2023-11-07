""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

import torchvision.models as models


class UpSample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, input_features, output_features, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#True
            self.conv = DoubleConv(input_features, output_features, input_features // 2)
        else:
            self.up = nn.ConvTranspose2d(input_features , input_features // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(input_features, output_features)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# class UpSample(nn.Sequential):
#     def __init__(self, input_features, output_features):
#         super(UpSample, self).__init__()        
#         self.convA = nn.Conv2d(input_features, output_features, kernel_size=3, stride=1, padding=1)
#         self.batchA = nn.BatchNorm2d(output_features)
#         self.leakyreluA = nn.LeakyReLU(0.5)
#         self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
#         self.batchB = nn.BatchNorm2d(output_features)
#         self.leakyreluB = nn.LeakyReLU(0.5)

#     def forward(self, x):
#         x = F.interpolate(x, size=[x.size(2)*2, x.size(3)*2], mode='bilinear', align_corners=True)
#         return self.leakyreluB( self.convB( self.leakyreluA(self.convA(x) )))

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,decoder=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_features = 2048
        self.resolution = 640
        self.resize = 80
        self.bilinear = bilinear
        self.decoder=decoder
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.down6 = Down(1024, 2048)

        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        if decoder:
            
            self.deconv1 = nn.Conv2d(self.input_features , self.input_features , kernel_size=1, stride=1, padding=0)
            self.up1 = UpSample(input_features=self.input_features  , output_features=self.input_features //2)
            self.up2 = UpSample(input_features=self.input_features //2,  output_features=self.input_features //4)
            self.up3 = UpSample(input_features=self.input_features //4 ,  output_features=self.input_features //8)
            self.up4 = UpSample(input_features=self.input_features //8,  output_features=self.input_features //16)
            self.up5 = UpSample(input_features=self.input_features //16,  output_features=self.input_features //32)
            self.up6 = UpSample(input_features=self.input_features //32,  output_features=self.input_features //64)
            
            self.deconv2 = nn.Conv2d(self.input_features //64, self.n_channels, kernel_size=3, stride=1, padding=1)

            # self.up1 = Up(2048, 2048 // factor, bilinear)
            # self.up2 = Up(2048, 1024 // factor, bilinear)
            # self.up3 = Up(1024, 512 // factor, bilinear)
            # # self.up4 = Up(512, 256, bilinear)
            # self.outc = OutConv(256, n_channels)           
            # self.up1 = Up(1024, 512 // factor, bilinear)
            # self.up2 = Up(512, 256 // factor, bilinear)
            # self.up3 = Up(256, 128 // factor, bilinear)
            # self.up4 = Up(128, 64, bilinear)
            # self.outc = OutConv(64, n_channels)
            # Linear layer
        # self.classifier1 = nn.Linear(4096, 2048)
        self.classifier1 = nn.Linear(2048, 1024)
        self.classifier2 = nn.Linear(1024, 512)
        self.classifier3 = nn.Linear(512, 256)
        self.classifier4 = nn.Linear(256, n_classes)
        #self.classifier5 = nn.Linear(128, n_classes)
         #self.classifier5 = nn.Linear(256, n_classes)
        self.classifier_activation = nn.ReLU(inplace=False)


    def forward(self, inputs):
        p1 =  inputs[:,:,0:self.resolution//2,0:self.resolution//2]
        p2 =  inputs[:,:,0:self.resolution//2,self.resolution//2:self.resolution]
        p3 =  inputs[:,:,self.resolution//2:self.resolution,0:self.resolution//2]
        p4 =  inputs[:,:,self.resolution//2:self.resolution,self.resolution//2:self.resolution]

        x1 = self.inc(p1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)

        y1 = self.inc(p2)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        # y5 = self.down4(y4)


        v1 = self.inc(p3)
        v2 = self.down1(v1)
        v3 = self.down2(v2)
        v4 = self.down3(v3)
        # v5 = self.down4(v4)


        z1 = self.inc(p4)
        z2 = self.down1(z1)
        z3 = self.down2(z2)
        z4 = self.down3(z3)
        # z5 = self.down4(z4)


        # x = self.up1(torch.cat([x5,y5,v5,z5],dim=1), torch.cat([x4,y4,v4,z4],dim=1))
        # x = self.up2(x, torch.cat([x3,y3,v3,z3],dim=1))
        # x = self.up3(x, torch.        m1 = x3+y3+v3+z3

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # if self.decoder:
        #     x = self.deconv1()
        #     x = self.up1(x5, x4)
        #     x = self.up2(x, x3)
        #     x = self.up3(x, x2)
        #     x = self.up4(x, x1)
        #     logits = self.outc(x)
        # else:
        #     logits = x

        marge = torch.FloatTensor(x3.size(0),256,self.resize,self.resize).cuda()
        marge[:,:,0:self.resize//2,0:self.resize//2] =  x4
        marge[:,:,0:self.resize//2,self.resize//2:self.resize] = y4
        marge[:,:,self.resize//2:self.resize,0:self.resize//2] = v4
        marge[:,:,self.resize//2:self.resize,self.resize//2:self.resize] = z4
        
        
        g1 = self.down4(marge)
        g2 = self.down5(g1)
        g3 = self.down6(g2)

        # decoder
        x = self.deconv1(F.relu(g3))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        logits = self.deconv2(x)

        
        out = self.classifier_activation(g3)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier1(out)
        out = self.classifier2(out)
        out = self.classifier3(out)
        out = self.classifier4(out)
        #out = self.classifier5(out)
         #out = self.classifier5(out)

        # logits = F.interpolate(logits, size=(640, 640), mode='bilinear')

        return logits, out
        # return out


        # return nn.functional.interpolate(logits, scale_factor=1, mode='bilinear', align_corners=True)

##########################################################################

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UNet1(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

##########################################################################

class VGGBigBlock(nn.Module):
    def __init__(self, in_channels, middle_channels1, middle_channels2, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels1)
        self.conv2 = nn.Conv2d(middle_channels1, middle_channels2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels2)
        self.conv3 = nn.Conv2d(middle_channels2, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

####################################################################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

####################################################################

class SegNet(nn.Module):

    
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.outc = OutConv(64, n_classes)

        # Linear layer
        self.classifier = nn.Linear(512, 2)

        # For the encoder

        # channels = [[64, 128, 256, 512, 512], [512, 256, 128, 64, 64]]
        nb_filter = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool = nn.MaxUnpool2d(2,2)
        
        self.conv0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2 = VGGBigBlock(nb_filter[1], nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3 = VGGBigBlock(nb_filter[2], nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv4 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[3])

        self.conv5 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv6 = VGGBigBlock(nb_filter[3], nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv7 = VGGBigBlock(nb_filter[2], nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv8 = VGGBlock(nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv9 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

        # Non trainable version of the VGG16 layers (without indices) :
        # self.inc = DoubleConv(n_channels, 64)
        # self.vgg16 = models.vgg16_bn(pretrained=True)
        # We have to remove the classifier
        # self.vgg16.classifier = Identity()


    def forward(self, input):

        # Encoder
        x0 = self.conv0(input)
        x1_pool, max_indices1 = self.pool(x0)
        x1 = self.conv1(x1_pool)

        x2_pool, max_indices2 = self.pool(x1)
        x2 = self.conv2(x2_pool)

        x3_pool, max_indices3 = self.pool(x2)
        x3 = self.conv3(x3_pool)

        x4_pool, max_indices4 = self.pool(x3)
        x4 = self.conv4(x4_pool)

        x5, max_indices5 = self.pool(x4)

        # Decoder
        x6 = self.conv5(self.unpool(x5, max_indices5))
        x7 = self.conv6(self.unpool(x6, max_indices4))
        x8 = self.conv7(self.unpool(x7, max_indices3))
        x9 = self.conv8(self.unpool(x8, max_indices2))
        x10 = self.conv9(self.unpool(x9, max_indices1))

        #output = self.final(x10)
        logits = self.outc(x10)

        # Classifier
        out = F.relu(x5, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        #print("out=",out.shape)
        out = self.classifier(out)

        return logits, out