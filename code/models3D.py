import torch
import torch.nn as nn
from monai.networks.nets import ResNet, ViT

class ResnetFeatureExtractor(nn.Module):
    """
    Resnet feature extractor. ResNet34 until stage 3.
    """
    def __init__(self, resnet_config):
        super().__init__()
        resnet = ResNet(**resnet_config)
                
        # Define the architecture - resnet until stage 3
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.act,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
            
    def forward(self, x):
        return self.features(x)
    
    def get_output_shape(self, input_shape):
        model_device = next(self.parameters()).device
        dummy_input = torch.randn(input_shape).to(model_device)
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
    
    def get_output_shape(self, input_shape):
        model_device = next(self.parameters()).device
        dummy_input = torch.randn(input_shape).to(model_device)
        self.eval()
        with torch.inference_mode():
            output = self.forward(dummy_input)
        self.train()
        return output.shape
    
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
    
    def get_output_shape(self, input_shape):
        model_device = next(self.parameters()).device
        dummy_input = torch.randn(input_shape).to(model_device)
        self.eval()
        with torch.inference_mode():
            output = self.forward(dummy_input)
        self.train()
        return output.shape
    
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    vit_config = {
    'in_channels': 256,
    'img_size': (6,7,6),
    'patch_size': (1,1,1),  
    'num_heads': 8,
    'hidden_size': 504,
    'mlp_dim': 2016,
    'num_layers': 7,
    'proj_type': 'perceptron',
    'pos_embed_type': 'sincos',
    'classification': True,
    'num_classes': 1,
    'dropout_rate': 0.0,
    'spatial_dims': 3,
    'post_activation': 'none',
    'qkv_bias': False,
    'save_attn': False
    }
    
    purevit_config = {
    'in_channels': 1,
    'img_size': (96,112,96),
    'patch_size': (16,16,16),
    'num_heads': 8,
    'hidden_size': 504,
    'mlp_dim': 2016,
    'num_layers': 7,
    'proj_type': 'perceptron',
    'pos_embed_type': 'sincos',
    'classification': True,
    'num_classes': 1,
    'dropout_rate': 0.0,
    'spatial_dims': 3,
    'post_activation': 'none',
    'qkv_bias': False,
    'save_attn': False
    }
    
    use_model = "hybrid" # Choose model: "hybrid", "resnet_extractor", "purevit"
    
    if use_model == "resnet_extractor" or use_model == "hybrid":
        print("\nExpected input shape is (91, 109, 91).")
        apply_padding = False
        input_shape = (1, 1, 91, 109, 91) 
    elif use_model == "purevit":
        print("\nExpected input shape is (96, 112, 96).")
        apply_padding = True
        input_shape = (1, 1, 96, 112, 96)
    else:
        raise ValueError("Invalid model choice. Use 'hybrid' or 'purevit'.")
    

    if use_model == "resnet_extractor":
        print("Using ResNet Feature Extractor.")
        model = ResnetFeatureExtractor(resnet_config=resnet_config).to(device)
        output_shape = model.get_output_shape(input_shape)
    elif use_model == "purevit":
        print("Using Pure ViT.")
        model = pureViT(vit_config=purevit_config).to(device)
        output_shape = model.get_output_shape(input_shape)
    elif use_model == "hybrid":
        print("Using Hybrid model.")
        resnet_extractor = ResnetFeatureExtractor(resnet_config=resnet_config)
        model = ResNetViT(resnet_feature_extractor=resnet_extractor, vit_config=vit_config).to(device)
        output_shape = model.get_output_shape(input_shape)
    else:
        raise ValueError("Invalid model choice. Use 'resnet_extractor' or 'purevit'.")
    
    print(f"Output shape: {output_shape}")
    
    
            