import torch
import torch.nn as nn
from monai.networks.nets import ResNet, ViT

class ResnetFeatureExtractor(nn.Module):
    """
    Resnet feature extractor. ResNet34 until stage 3.
    """
    def __init__(self, resnet_config, device):
        super().__init__()
        self.device = device
        resnet = ResNet(**resnet_config).to(device)
                
        # Define the architecture - resnet until stage 3
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.act,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        ).to(device)
            
    def forward(self, x):
        return self.features(x)
    
    def get_output_shape(self, input_shape):
        dummy_input = torch.randn(input_shape).to(self.device)
        self.eval()
        with torch.inference_mode():
            output = self.forward(dummy_input)
        self.train()
        return output.shape
    
class ResNetViT(nn.Module):
    """
    ResNet Feature Extractor + ViT model.
    """
    def __init__(self, resnet_feature_extractor, vit_config):
        super().__init__()
        self.resnet_feature_extractor = resnet_feature_extractor
        self.vit = ViT(**vit_config)
    
    def forward(self, x):
        x = self.resnet_feature_extractor(x)
        x = self.vit(x)
        # vit returns a tuple (loss is the first element)
        if isinstance(x, tuple):
            x = x[0]
        else:
            pass
        return x
    
class pureViT(nn.Module):
    """
    Pure ViT model.
    """
    def __init__(self, vit_config):
        super().__init__()
        self.vit = ViT(**vit_config)
    
    def forward(self, x):
        x = self.vit(x)
        # vit returns a tuple (loss is the first element)
        if isinstance(x, tuple):
            x = x[0]
        else:
            raise ValueError("Expected output to be a tuple, but got: {}".format(type(x)))
        return x
    
if __name__ == '__main__':
    resnet_config = {
        'block': 'basic',
        'layers': [3, 4, 6, 3],
        'block_inplanes': [64, 128, 256, 512],
        'spatial_dims': 3,
        'n_input_channels': 1,
        'conv1_t_stride': 2,
        'num_classes': 1, 
        'shortcut_type': 'B',
        'bias_downsample': True
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    feature_extractor = ResnetFeatureExtractor(resnet_config=resnet_config, device=device)
    input_shape = (1, 1, 91, 109, 91) 
    output_shape = feature_extractor.get_output_shape(input_shape)
    print(f"Output shape: {output_shape}")
    
    
            