import torch
import torch.nn as nn
from monai.networks.nets import ResNet, ViT

class ResnetFeatureExtractor(nn.Module):
    """
    Resnet feature extractor. ResNet34 until stage 3.
    """
    def __init__(self, resnet_config, device, trained_path=None, use_pretrained=False):
        super().__init__()
        self.device = device
        resnet = ResNet(**resnet_config).to(device)
                
        if use_pretrained:
            print("Loading pretrained weights...")
            if trained_path is None:
                raise ValueError("trained_path must be provided if use_pretrained is True")
            else:
                resnet.load_state_dict(torch.load(trained_path, weights_only=True, map_location=self.device))
        
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
            
            # Freeze the weights
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            print("Training from scratch...")
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
    
if __name__ == '__main__':
    resnet_config = {
        'block': 'basic',
        'layers': [3, 4, 6, 3],
        'block_inplanes': [64, 128, 256, 512],
        'spatial_dims': 3,
        'n_input_channels': 1,
        'conv1_t_stride': 2,
        'num_classes': 1, # doesn't matter
        'shortcut_type': 'B',
        'bias_downsample': False
    }
    
    use_pretrained = False
    if use_pretrained:
        resnet_path = "/home/diogommiranda/tese/outputs/torch/full_brain/fixed_lr/CROSS_VALIDATION/saved_models/LR=1.0e-05_WD=1e-04 (sgd)/model.pth"
    else:
        resnet_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    feature_extractor = ResnetFeatureExtractor(resnet_config=resnet_config, device=device, trained_path=resnet_path, use_pretrained=use_pretrained)
    input_shape = (1, 1, 91, 109, 91) 
    output_shape = feature_extractor.get_output_shape(input_shape)
    print(f"Output shape: {output_shape}")
    
    
            