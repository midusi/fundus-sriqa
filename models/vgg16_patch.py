import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import models
from models.unet.unet_parts import *


def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = False) -> None:
        """

        Args:
            num_filters:
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class Interpolate(nn.Module):
    def __init__(
        self,
        size: int = None,
        scale_factor: int = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class DecoderBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs,skip):
        print("forward inputs = ",inputs.size(2))
        print("forward skip = ",skip.size(2))
        if inputs.size(2) == skip.size(2):
            x = inputs
        else:
            x = self.up(inputs)
            x = self.up(x)
        print('************* x ',x.shape)
        print('************* skip ',skip.shape)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
    
class decoder_block_without_skip(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c, out_c)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

# class UNet16(nn.Module):
#     def __init__(
#         self,
#         in_channels:int=3,
#         num_classes: int = 3,
#         num_filters: int = 32,
#         pretrained: bool = True,
#         is_deconv: bool = False,
#     ):
#         """

#         Args:
#             num_classes:
#             num_filters:
#             pretrained:
#                 False - no pre-trained network used
#                 True - encoder pre-trained with VGG16
#             is_deconv:
#                 False: bilinear interpolation is used in decoder
#                 True: deconvolution is used in decoder
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.inc = DoubleConv(in_channels, 32)
#         self.resolution = 640
#         self.resize = 80

#         self.pool = nn.MaxPool2d(2, 2)

#         self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

#         self.relu = nn.ReLU(inplace=True)

#         self.conv1 = nn.Sequential(
#             self.encoder[0], self.relu, self.encoder[2], self.relu,
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         )

#         self.conv2 = nn.Sequential(
#             self.encoder[5], self.relu, self.encoder[7], self.relu,
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )

#         self.conv3 = nn.Sequential(
#             self.encoder[10],
#             self.relu,
#             self.encoder[12],
#             self.relu,
#             self.encoder[14],
#             self.relu,
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )

#         self.conv4 = nn.Sequential(
#             self.encoder[17],
#             self.relu,
#             self.encoder[19],
#             self.relu,
#             self.encoder[21],
#             self.relu,
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )

#         self.conv5 = nn.Sequential(
#             self.encoder[24],
#             self.relu,
#             self.encoder[26],
#             self.relu,
#             self.encoder[28],
#             self.relu,
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         # We dont use this decoder
#         self.center = DecoderBlockV2(
#             512, num_filters * 8 * 2, num_filters * 8, is_deconv
#         )

#         self.dec5 = DecoderBlockV2(
#             512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
#         )
#         self.dec4 = DecoderBlockV2(
#             512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
#         )
#         self.dec3 = DecoderBlockV2(
#             256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv
#         )
#         self.dec2 = DecoderBlockV2(
#             128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv
#         )
#         self.dec1 = ConvRelu(64 + num_filters, num_filters)
#         self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

#          # Linear layer
#         self.classifier1 = nn.Linear(512, 256)
#         self.classifier2 = nn.Linear(256, 128)
#         self.classifier3 = nn.Linear(128, 64)
#         self.classifier4 = nn.Linear(64, self.num_classes)


# #  self.down1 = Down(32, 64)
# #         self.down2 = Down(64, 128)
# #         self.down3 = Down(128, 256)
# #         self.down4 = Down(256, 512)
#         """ Bottleneck """
#         self.b = conv_block(256,256)


#         self.d1=decoder_block(256,128)     
#         self.d2=decoder_block(128,64)   
#         self.d3=decoder_block(64,3)   

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:

#         #print('************************************************************')

#         p1 =  inputs[:,:,0:self.resolution//2,0:self.resolution//2]
#         p2 =  inputs[:,:,0:self.resolution//2,self.resolution//2:self.resolution]
#         p3 =  inputs[:,:,self.resolution//2:self.resolution,0:self.resolution//2]
#         p4 =  inputs[:,:,self.resolution//2:self.resolution,self.resolution//2:self.resolution]

#         # x1 = self.inc(p1)
      
#         x2 = self.conv1(p1)
#         x3 = self.conv2(x2)
#         x4 = self.conv3(x3)

#         '''   strat decoder (reconstructions)'''
        
#         d_x=self.b(x4)
#         d1_x=self.d1(d_x,x3)
        
#         d2_x=self.d2(d1_x,x2)
        
#         d3_x=self.d3(d2_x,p1)
       

#         # y1 = self.inc(p2)
#         y2 = self.conv1(p2)
        
#         y3 = self.conv2(y2)
#         y4 = self.conv3(y3)
#         '''   strat decoder (reconstructions)'''
#         d_y=self.b(y4)
#         d1_y=self.d1(d_y,y3)
        
#         d2_y=self.d2(d1_y,y2)
        
#         d3_y=self.d3(d2_y,p2)
       
#         # v1 = self.inc(p3)
#         v2 = self.conv1(p3)
        
#         v3 = self.conv2(v2)
#         v4 = self.conv3(v3)
#         '''   strat decoder (reconstructions)'''
#         d_v=self.b(v4)
#         d1_v=self.d1(d_v,v3)
        
#         d2_v=self.d2(d1_v,v2)
        
#         d3_v=self.d3(d2_v,p3)

#         # z1 = self.inc(p4)
#         z2 = self.conv1(p4)
       
#         z3 = self.conv2(z2)
#         z4 = self.conv3(z3)

#         '''   strat decoder (reconstructions)'''
#         d_z=self.b(z4)
#         d1_z=self.d1(d_z,z3)
        
#         d2_z=self.d2(d1_z,z2)
        
#         d3_z=self.d3(d2_z,p4)

#         # conv1 = self.conv1(x)
#         # conv2 = self.conv2(self.pool(conv1))
#         # conv3 = self.conv3(self.pool(conv2))
        

        
#         marge = torch.FloatTensor(x4.size(0),256,self.resize,self.resize).cuda()
        
       
#         marge[:,:,0:self.resize//2,0:self.resize//2] =  x4
#         marge[:,:,0:self.resize//2,self.resize//2:self.resize] = y4
#         marge[:,:,self.resize//2:self.resize,0:self.resize//2] = v4
#         marge[:,:,self.resize//2:self.resize,self.resize//2:self.resize] = z4





#         conv4 = self.conv4(self.pool(marge))
#         conv5 = self.conv5(self.pool(conv4))
        

#         # Classifier
#         out = F.relu(self.pool(conv5), inplace=False)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier1(out)
#         out = self.classifier2(out)
#         out = self.classifier3(out)
#         out = self.classifier4(out)

#         # center = self.center(self.pool(conv5))
#         # dec5 = self.dec5(torch.cat([center, conv5], 1))

#         # dec4 = self.dec4(torch.cat([dec5, conv4], 1))



#         # dec3 = self.dec3(torch.cat([dec4, conv3], 1))
#         # dec2 = self.dec2(torch.cat([dec3, conv2], 1))
#         # dec1 = self.dec1(torch.cat([dec2, conv1], 1))

#         original_image_batch_size=640
#         final_decoder = torch.FloatTensor(x3.size(0),3,original_image_batch_size,original_image_batch_size).cuda()
       
#         final_decoder[:,:,0:original_image_batch_size//2,0:original_image_batch_size//2] =  d3_x
#         final_decoder[:,:,0:original_image_batch_size//2,original_image_batch_size//2:original_image_batch_size] = d3_y
#         final_decoder[:,:,original_image_batch_size//2:original_image_batch_size,0:original_image_batch_size//2] = d3_v
#         final_decoder[:,:,original_image_batch_size//2:original_image_batch_size,original_image_batch_size//2:original_image_batch_size] = d3_z
        
#         return final_decoder, out

from iterative_normalization import IterNormRotation as cw_layer

class UNet16(nn.Module):
    def __init__(
        self,
        in_channels:int=3,
        num_classes: int = 3,
        num_filters: int = 32,
        pretrained: bool = True,
        is_deconv: bool = False,
    ):
        """

        Args:
            num_classes:
            num_filters:
            pretrained:
                False - no pre-trained network used
                True - encoder pre-trained with VGG16
            is_deconv:
                False: bilinear interpolation is used in decoder
                True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.inc = DoubleConv(in_channels, 32)
        self.resolution =1280#640#1280#320 #480#1920 #1280 #640
        self.resize1 =160#80#160#40 #60#240 #160 #80
        self.resize2 =320#160#320#40 #60#240 #160 #80
        self.resize3 =640#320#640#40 #60#240 #160 #80

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )

        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            # cw_layer(self.encoder[17].out_channels),
            self.relu,
            self.encoder[19],
            # cw_layer(self.encoder[19].out_channels),
            self.relu,
            self.encoder[21],
            cw_layer(self.encoder[21].out_channels),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # We dont use this decoder
        self.center = DecoderBlockV2(
            512, num_filters * 8 * 2, num_filters * 8, is_deconv
        )

        self.dec5 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec4 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec3 = DecoderBlockV2(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv
        )
        self.dec2 = DecoderBlockV2(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv
        )
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

         # Linear layer
        self.classifier1 = nn.Linear(512, 256)
        self.classifier2 = nn.Linear(256, 128)
        self.classifier3 = nn.Linear(128, 64)
        self.classifier4 = nn.Linear(64, self.num_classes)


#  self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 256)
#         self.down4 = Down(256, 512)
        """ Bottleneck """
        self.b = conv_block(512,512)


        self.d1 = decoder_block_without_skip(512,512)  
        self.d2 = decoder_block_without_skip(512,512)     
        self.d3 = decoder_block_without_skip(512,256)   
        self.d4 = decoder_block_without_skip(256,128)   
        self.d5 = decoder_block_without_skip(128,64)   
        self.d6 = decoder_block_without_skip(64,32)  
        self.d7 = decoder_block_without_skip(32,3)  

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        self.conv4[5].mode = mode
    
    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        self.conv4[5].update_rotation_matrix()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        #print('************************************************************')

        p1 =  inputs[:,:,0:self.resolution//2,0:self.resolution//2]
        p2 =  inputs[:,:,0:self.resolution//2,self.resolution//2:self.resolution]
        p3 =  inputs[:,:,self.resolution//2:self.resolution,0:self.resolution//2]
        p4 =  inputs[:,:,self.resolution//2:self.resolution,self.resolution//2:self.resolution]

        # x1 = self.inc(p1)
      
        x2 = self.conv1(p1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)

        '''   strat decoder (reconstructions)'''
        
        # d_x=self.b(x4)
        # d1_x=self.d1(d_x,x3)
        
        # d2_x=self.d2(d1_x,x2)
        
        # d3_x=self.d3(d2_x,p1)
       

        # y1 = self.inc(p2)
        y2 = self.conv1(p2)
        
        y3 = self.conv2(y2)
        y4 = self.conv3(y3)
        '''   strat decoder (reconstructions)'''
        # d_y=self.b(y4)
        # d1_y=self.d1(d_y,y3)
        
        # d2_y=self.d2(d1_y,y2)
        
        # d3_y=self.d3(d2_y,p2)
       
        # v1 = self.inc(p3)
        v2 = self.conv1(p3)
        
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        '''   strat decoder (reconstructions)'''
        # d_v=self.b(v4)
        # d1_v=self.d1(d_v,v3)
        
        # d2_v=self.d2(d1_v,v2)
        
        # d3_v=self.d3(d2_v,p3)

        # z1 = self.inc(p4)
        z2 = self.conv1(p4)
       
        z3 = self.conv2(z2)
        z4 = self.conv3(z3)

        '''   strat decoder (reconstructions)'''
        # d_z=self.b(z4)
        # d1_z=self.d1(d_z,z3)
        
        # d2_z=self.d2(d1_z,z2)
        
        # d3_z=self.d3(d2_z,p4)

        # conv1 = self.conv1(x)
        # conv2 = self.conv2(self.pool(conv1))
        # conv3 = self.conv3(self.pool(conv2))
        

        
        # # Change
        # marge = torch.cat([x4,y4,v4,z4],dim=1)
        # convA = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1).cuda()
        # marge = convA(marge)

       

        

       
        #marge4 = torch.FloatTensor(x4.size(0),256,self.resize1,self.resize1).cuda()
        size_marge4 = (x4.size(0),256,self.resize1,self.resize1)
        marge4=torch.empty(size_marge4,dtype=x4.dtype,device=x4.device)
        marge4[:,:,0:self.resize1//2,0:self.resize1//2] =  x4
        marge4[:,:,0:self.resize1//2,self.resize1//2:self.resize1] = y4
        marge4[:,:,self.resize1//2:self.resize1,0:self.resize1//2] = v4
        marge4[:,:,self.resize1//2:self.resize1,self.resize1//2:self.resize1] = z4


        marge3 = torch.FloatTensor(x3.size(0),128,self.resize2,self.resize2).cuda()
        marge3[:,:,0:self.resize2//2,0:self.resize2//2] =  x3
        marge3[:,:,0:self.resize2//2,self.resize2//2:self.resize2] = y3
        marge3[:,:,self.resize2//2:self.resize2,0:self.resize2//2] = v3
        marge3[:,:,self.resize2//2:self.resize2,self.resize2//2:self.resize2] = z3

        marge2 = torch.FloatTensor(x2.size(0),64,self.resize3,self.resize3).cuda()
        marge2[:,:,0:self.resize3//2,0:self.resize3//2] =  x2
        marge2[:,:,0:self.resize3//2,self.resize3//2:self.resize3] = y2
        marge2[:,:,self.resize3//2:self.resize3,0:self.resize3//2] = v2
        marge2[:,:,self.resize3//2:self.resize3,self.resize3//2:self.resize3] = z2
       

        conv4 = self.conv4(self.pool(marge4))
        conv5 = self.conv5(self.pool(conv4))


        # Classifier
        out = F.relu(self.pool(conv5), inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier1(out)
        out = self.classifier2(out)
        out = self.classifier3(out)
        out = self.classifier4(out)



        dec0 = self.b(conv5)
        dec1 = self.d1(dec0)
        dec2 = self.d2(dec1)
        dec3 = self.d3(dec2)
        dec4 = self.d4(dec3)
        dec5 = self.d5(dec4)
        dec6 = self.d6(dec5)
        recoimage = self.d7(dec6)

        # print("########### dec1 = ",dec1.shape)
        # print("########### dec2 = ",dec2.shape)
        # print("########### dec3 = ",dec3.shape)
        # print("########### dec4 = ",dec4.shape)
        # print("########### dec5 = ",dec5.shape)
        # print("########### reconstract = ",recoimage.shape)

        # center = self.center(self.pool(conv5))
        # dec5 = self.dec5(torch.cat([center, conv5], 1))

        # dec4 = self.dec4(torch.cat([dec5, conv4], 1))



        # dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        # dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        # dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        # original_image_batch_size=self.resolution
        # final_decoder = torch.FloatTensor(x3.size(0),3,original_image_batch_size,original_image_batch_size).cuda()
       
        # final_decoder[:,:,0:original_image_batch_size//2,0:original_image_batch_size//2] =  d3_x
        # final_decoder[:,:,0:original_image_batch_size//2,original_image_batch_size//2:original_image_batch_size] = d3_y
        # final_decoder[:,:,original_image_batch_size//2:original_image_batch_size,0:original_image_batch_size//2] = d3_v
        # final_decoder[:,:,original_image_batch_size//2:original_image_batch_size,original_image_batch_size//2:original_image_batch_size] = d3_z
        
        return recoimage, out

