import models
from transformers import AutoFeatureExtractor, ViTForImageClassification, SwinForImageClassification

def define_resnet(pooling, size, pretrained = True, gamma = None):
    model = models.ResNet(pooling,
                            size, 
                            pretrained=pretrained, 
                            gamma=gamma)

    model.requires_grad_(True)
    return model   


def define_efficientnet(pooling, size, pretrained = True, gamma = None, dropout = 0.):
    model = models.EfficientNet(pooling,
                                size, 
                                pretrained=pretrained, 
                                gamma=gamma, 
                                dropout=dropout)
    model.requires_grad_(True)
    return model       

def define_mobilenet(pooling, pretrained = True, gamma = None, dropout = 0.):
    model = models.MobileNet(pooling, 
                            pretrained=pretrained, 
                            gamma=gamma, 
                            dropout=dropout)
    model.requires_grad_(True)
    return model    
              
        
def define_vgg(pretrained = True, dropout = 0.):
    model = models.VGG(pretrained=pretrained, dropout=dropout)    
    model.requires_grad_(True)
    return model   


def define_deits():
    feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-small-patch16-224')
    model = ViTForImageClassification.from_pretrained('facebook/deit-small-patch16-224', num_labels=1, ignore_mismatched_sizes=True)
    return model, feature_extractor

def define_vit():
    feature_extractor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=1, ignore_mismatched_sizes=True)
    return model, feature_extractor
    

def define_swintiny():
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=1, ignore_mismatched_sizes=True)
    return model, feature_extractor
    
def define_swinbase():
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224", num_labels=1, ignore_mismatched_sizes=True)
    return model, feature_extractor