
import torch
from .resnet_model_big import ResNet101, ResNet152
from .resnet import ResNet, DeepResNet
from .other_models import SegNetWithTorchModel, UNetWithDenseNetEncoder,UNetWithResNetEncoder
from .vgg16_patch import  UNet16
from .unet import UNet
from .FGRNET import UNet16V2

def create_model(model_name,device:torch.DeviceObjType, num_classes=3):
    if model_name == "ResNet":
      model = ResNet(3,num_classes)
      encoder = model.resnet
    elif model_name == "SegnetTorchModel":
      model = SegNetWithTorchModel(3,num_classes)
      # model.encoder = torch.nn.Sequential(model.vgg16,View((-1,512,10,10)))
      model.encoder_reshape=(-1,512,10,10)
      encoder = model.vgg16
    elif model_name == "SegnetTorchModel/Contrast":
      model = SegNetWithTorchModel(3,num_classes)
      model.encoder_reshape=(-1,512,10,10)
      encoder = model.vgg16
    elif model_name == "ResNet/Deep":
      model = DeepResNet(3,3)
      encoder = model.resnet
    elif model_name == "ResNet/DeepSSIM":
      model = DeepResNet(3,3)
      encoder = model.resnet
    elif model_name == "UNetWithDenseNetEncoder":
      model = UNetWithDenseNetEncoder(3,2)#3
      encoder = model.encoder
    elif model_name == "ResNet/101":
      model = ResNet101(3,2)#3
      encoder = model.resnet
    elif model_name == "ResNet/152":
      model = ResNet152(3,3)
      encoder = model.resnet
    elif model_name=="UNet":
      model = UNet(3,num_classes).to(device)
      encoder = model.down4
    elif model_name=="UNetNoDecoder":
      model = UNet(3,num_classes,decoder=False).to(device)
      encoder = model.down4
    elif model_name == "UNetWithResNetEncoder":
      model = UNetWithResNetEncoder(3,num_classes).to(device)
      encoder = ""
    elif model_name == "vgg16patch":
      model = UNet16(3,num_classes=num_classes)
      encoder = model.encoder[28] #conv4
    elif model_name == "FGRNET":
      model = UNet16V2(num_classes)
      encoder = model.conv5 #encoder[28]
    else:
      print("Invalid model name {model_name}")
      raise Exception()
    model.name=model_name
    return model.to(device),encoder 