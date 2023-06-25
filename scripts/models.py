import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from einops import rearrange
from pooling import *
import attentions


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def save_model_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_model_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        

class VGG(BaseModel):
    def __init__(self, dropout = .0, pretrained = True):
        super().__init__()
        self.backbone = models.vgg11_bn(pretrained=pretrained)
            
        self.backbone.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096//4),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096//4, 1),
        )

    def forward(self,x):
        return self.backbone(x)  
    
    
class ResNet(BaseModel):
    def __init__(self, pooling, size, gamma=5, pretrained = True):
        super(ResNet, self).__init__()
        if size=='18':
            self.backbone = models.resnet18(pretrained=pretrained) 
            self.backbone.fc = nn.Linear(512, 1)
            print('resnet18')
        elif size=='34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.backbone.fc = nn.Linear(512, 1)
            print('resnet34')
        elif size=='50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone.fc = nn.Linear(2048, 1)
            print('resnet50')

        else:
            raise RuntimeError()
        
        if pooling == 'LSE':
            self.backbone.avgpool = LogSumExpPool(gamma)
            
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x):
        B,C,H,W = x.size()
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        b, c, h, w = x.shape
        cam = rearrange(x, "B C H W -> B (H W) C")
        cam = self.backbone.fc(cam)
        cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return cam, x
    
    
class EfficientNet(BaseModel):
    def __init__(self, pooling, size, gamma=5, dropout = .0, pretrained = True):
        super().__init__()
        if size == 'b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            print('efficientnet_b0')
        elif size== 'b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            print('efficientnet_b1')
        else:
            raise RuntimeError()
        
        if pooling == 'LSE':
            self.backbone.avgpool = LogSumExpPool(gamma)
            
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1280, 1),
        )

    def forward(self,x):
        return self.backbone(x)  
    
    def predict(self, x):
        B,C,H,W = x.size()
        x = self.backbone.features(x)
        b, c, h, w = x.shape
        cam = rearrange(x, "B C H W -> B (H W) C")
        cam = self.backbone.classifier(cam)
        cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.backbone.classifier(x)
        return cam, x
              
               
class MobileNet(BaseModel):
    def __init__(self, pooling, gamma=5, dropout = .0, pretrained = True):    
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        if pooling == 'LSE':
            self.backbone.avgpool = LogSumExpPool(gamma)
    
        self.backbone.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, 1),
        )
        
    def forward(self,x):
        return self.backbone(x)
    
    def predict(self, x):
        B,C,H,W = x.size()
        x = self.backbone.features(x)
        b, c, h, w = x.shape
        cam = rearrange(x, "B C H W -> B (H W) C")
        cam = self.backbone.classifier(cam)
        cam = F.upsample(rearrange(cam, "B (H W) C -> B C H W", C=1,H=h,W=w), size=(H,W), align_corners=True, mode='bilinear')
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.backbone.classifier(x)
        return cam, x
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,*args, kernel_size=3, stride=1, padding=1, **kwargs):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels, 
                                out_channels, 
                                (kernel_size, kernel_size), 
                                stride = (stride,stride), 
                                padding = (padding, padding), 
                                bias=False), 
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU())
    
    def forward(self, x):
        return self.layers(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, dim_in, *args, dim_out=None, kernel_size=3, stride=1, padding=1, **kwargs):
        super(ResnetBlock, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(dim_out)
        
        self.identity = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(dim_out)) if dim_in!=dim_out else nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.bn2(z)
        identity = self.identity(x)
        z+=identity
        z = self.relu(z)
        return z

class LocalMSA(nn.Module):
    def __init__(self, in_channels, heads, dim_head, inner_dim=None, window_Size=7, dropout=0.0):
        super().__init__()
        inner_dim = in_channels//2 if inner_dim is None else inner_dim
        self.conv = nn.Conv2d(in_channels, inner_dim, kernel_size=1)
        self.bn   = nn.BatchNorm2d(inner_dim)
        self.attn = attentions.LocalAttention(inner_dim, dim_out= in_channels, window_size=window_Size, heads=heads, dim_head=dim_head, dropout=dropout)
        
    def forward(self, x):
        out = self.bn(self.conv(x))
        out, _ = self.attn(out)
        return x + out
    

### Blocks

class Res(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dim_head, dropout) -> None:
        super().__init__()
        self.layers = ResnetBlock(dim_in=in_channels, dim_out=out_channels)
                                   
    def forward(self, x):
        return self.layers(x)  


class ResMsa(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dim_head, dropout) -> None:
        super().__init__()
        self.layers = nn.Sequential(ResnetBlock(dim_in=in_channels, dim_out=out_channels),
                                    LocalMSA(out_channels, heads, dim_head, dropout=dropout))
            
    def forward(self, x):
        return self.layers(x)  
      

# AlterNet-K base model
class AlterNetK(BaseModel):
    def __init__(self, layers_str, in_channels_str, out_channel_str, heads_str, head_dim, dropout=0.0):
        """_summary_

        Args:
            layers_str (str): string containing layer module names ex. 'ResMsa-ResMsa-ResMsa-ResMsa-ResMsa'
            in_channels_str (str): input dims for each stage ex. '32-64-128-128-128'
            out_channel_str (str): output dims for each stage ex. '64-128-128-128-128'
            heads_str (str): number for heads for MSA of each stage ex. '2-4-4-4-4'
            head_dim (int): 16
            dropout (float, optional): 0.1. Defaults to 0.0.

        """
        super().__init__()
        layers_str = layers_str.split('-')
        in_channel_list = list(map(lambda x:int(x), in_channels_str.split('-')))
        out_channel_list = list(map(lambda x:int(x), out_channel_str.split('-')))
        heads_list = list(map(lambda x:int(x), heads_str.split('-')))
        
        self.initial = ConvBlock(3,in_channel_list[0]) #224
        self.maxpool = nn.MaxPool2d(2,2)
        
        if len(in_channel_list)!= len(out_channel_list)!=len(heads_list):
            raise ValueError("number of in/out channels and heads mismatched")
        
        class_dict = {
                      'Res':Res,
                      'ResMsa':ResMsa
                      }
        
        layers = list()
    
        for layer, in_channels, out_channels, heads in zip(layers_str,in_channel_list, out_channel_list, heads_list):
            layers.append(class_dict[layer](in_channels, out_channels, heads, head_dim, dropout))
        
        self.layers = nn.ModuleList(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(out_channel_list[-1], 1)

    def forward(self, x):
        x = self.maxpool(self.initial(x))
        for layer in self.layers[:-1]:
            x = self.maxpool(layer(x))
        x = self.avgpool(self.layers[-1](x)).squeeze(-1).squeeze(-1)
        return self.classifier(x)

