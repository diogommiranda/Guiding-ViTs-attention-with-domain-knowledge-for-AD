import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import ResNet, ViT

class ResnetFeatureExtractor(nn.Module):
    """
    Resnet feature extractor. ResNet until stage 3.
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
    
class ResNetViT(nn.Module):
    """
    ResNet Feature Extractor + ViT model.
    """
    def __init__(self, resnet_feature_extractor, vit_config):
        super().__init__()
        self.resnet_feature_extractor = resnet_feature_extractor
        self.vit = ViT(**vit_config)
        self.attention_maps = []
    
    def forward(self, x, return_attention=None):
        self.attention_maps = []
        
        x = self.resnet_feature_extractor(x)
        x, hidden_states = self.vit(x)
    
        if return_attention == 'softmaxed' or return_attention == 'raw':
            self.extract_attention_maps(return_attention)
        
        return x, self.attention_maps
    
    def extract_attention_maps(self, return_attention):
        self.attention_maps = []
        if return_attention == 'softmaxed':
            for i, block in enumerate(self.vit.blocks):
                if hasattr(block.attn, 'att_mat') and block.attn.att_mat is not None:
                    attention_matrix = block.attn.att_mat.detach()
                    self.attention_maps.append(attention_matrix)
        elif return_attention == 'raw':
            for i, block in enumerate(self.vit.blocks):
                if hasattr(block.attn, 'att_logits_raw') and block.attn.att_logits_raw is not None:
                    attention_matrix = block.attn.att_logits_raw.detach()
                    self.attention_maps.append(attention_matrix)
    
    def get_attention_map(self, layer=None, head=None, average_heads=False):
        """
        Get specific attention map or averaged attention maps.
        
        Args:
            layer: Layer number (None for all layers)
            head: Attention head number (None for all heads)
            average_heads: Whether to average across attention heads
            
        Returns:
            Attention map(s)
        """
        if not self.attention_maps:
            raise ValueError("No attention maps available. Run forward() with return_attention=True first.")
        
        
        if layer is not None:
            if layer > len(self.attention_maps) or layer <= 0:
                raise ValueError(f"Layer {layer} out of range. Available layers: 1-{len(self.attention_maps)}")
            
            attn_map = self.attention_maps[layer-1]
            
            if head is not None:
                if head > attn_map.shape[1] or head <= 0:
                    raise ValueError(f"Head {head} out of range. Available heads: 1-{attn_map.shape[1]}")
                return attn_map[:, head-1, :, :]  # [batch_size, seq_len, seq_len]
            
            if average_heads:
                return attn_map.mean(dim=1)  # Average across heads: [batch_size, seq_len, seq_len]
            else:
                return attn_map  # [batch_size, num_heads, seq_len, seq_len]
        else:
            # Return all attention maps
            all_maps = []
            for attn_map in self.attention_maps:
                if average_heads:
                    attn_map = attn_map.mean(dim=1)
                all_maps.append(attn_map)
            return all_maps
        
    def visualize_attention_pattern(self, layer, head=None):
        """
        Create a simple text representation of attention patterns.
        
        Args:
            layer: Layer to visualize
            head: Specific head to visualize (None to average all heads)
        """
        if not self.attention_maps:
            raise ValueError("No attention maps available. Run forward() with return_attention=True first.")
        

        attn_map = self.get_attention_map(layer, head, average_heads=(head is None))
        
        # Take first sample in batch
        attn_sample = attn_map[0].cpu().numpy()
        
        print(f"Attention Pattern - Layer {layer}" + (f", Head {head}" if head is not None else " (averaged)"))
        print(f"Shape: {attn_sample.shape}")
        print(f"Min: {attn_sample.min():.4f}, Max: {attn_sample.max():.4f}")
        
        # Show a small portion of the attention matrix
        if attn_sample.shape[0] <= 10:
            print("Attention Matrix:")
            for i in range(attn_sample.shape[0]):
                row_str = " ".join([f"{val:.3f}" for val in attn_sample[i, :min(10, attn_sample.shape[1])]])
                print(f"Token {i:2d}: {row_str}")
        else:
            print("Attention Matrix (first 10x10):")
            for i in range(10):
                row_str = " ".join([f"{val:.3f}" for val in attn_sample[i, :10]])
                print(f"Token {i:2d}: {row_str}")
    
class pureViT(nn.Module):
    """
    Pure ViT model.
    """
    def __init__(self, vit_config):
        super().__init__()
        self.vit = ViT(**vit_config)
        self.attention_maps = []
    
    def forward(self, x, return_attention=None):
        self.attention_maps = []
        
        x, hidden_states = self.vit(x)
        
        if return_attention == 'softmaxed' or return_attention == 'raw':
            self.extract_attention_maps(return_attention)
            
        return x, self.attention_maps
        
    def extract_attention_maps(self, return_attention):
        self.attention_maps = []
        if return_attention == 'softmaxed':
            for i, block in enumerate(self.vit.blocks):
                if hasattr(block.attn, 'att_mat') and block.attn.att_mat is not None:
                    attention_matrix = block.attn.att_mat.detach()
                    self.attention_maps.append(attention_matrix)
        elif return_attention == 'raw':
            for i, block in enumerate(self.vit.blocks):
                if hasattr(block.attn, 'att_logits_raw') and block.attn.att_logits_raw is not None:
                    attention_matrix = block.attn.att_logits_raw.detach()
                    self.attention_maps.append(attention_matrix)
    
    def get_attention_map(self, layer=None, head=None, average_heads=False):
        """
        Get specific attention map or averaged attention maps.
        
        Args:
            layer: Layer number (None for all layers)
            head: Attention head number (None for all heads)
            average_heads: Whether to average across attention heads
            
        Returns:
            Attention map(s)
        """
        if not self.attention_maps:
            raise ValueError("No attention maps available. Run forward() with return_attention=True first.")
        
        if layer is not None:
            if layer > len(self.attention_maps) or layer <= 0:
                raise ValueError(f"Layer {layer} out of range. Available layers: 1-{len(self.attention_maps)}")
            
            attn_map = self.attention_maps[layer-1]
            
            if head is not None:
                if head > attn_map.shape[1] or head <= 0:
                    raise ValueError(f"Head {head} out of range. Available heads: 1-{attn_map.shape[1]}")
                return attn_map[:, head-1, :, :]  # [batch_size, seq_len, seq_len]
            
            if average_heads:
                return attn_map.mean(dim=1)  # Average across heads: [batch_size, seq_len, seq_len]
            else:
                return attn_map  # [batch_size, num_heads, seq_len, seq_len]
        else:
            # Return all attention maps
            all_maps = []
            for attn_map in self.attention_maps:
                if average_heads:
                    attn_map = attn_map.mean(dim=1)
                all_maps.append(attn_map)
            return all_maps
        
    def visualize_attention_pattern(self, layer, head=None):
        """
        Create a simple text representation of attention patterns.
        
        Args:
            layer: Layer to visualize
            head: Specific head to visualize (None to average all heads)
        """
        if not self.attention_maps:
            raise ValueError("No attention maps available. Run forward() with return_attention=True first.")
        

        attn_map = self.get_attention_map(layer, head, average_heads=(head is None))
        
        # Take first sample in batch
        attn_sample = attn_map[0].cpu().numpy()
        
        print(f"Attention Pattern - Layer {layer}" + (f", Head {head}" if head is not None else " (averaged)"))
        print(f"Shape: {attn_sample.shape}")
        print(f"Min: {attn_sample.min():.4f}, Max: {attn_sample.max():.4f}")
        
        # Show a small portion of the attention matrix
        if attn_sample.shape[0] <= 10:
            print("Attention Matrix:")
            for i in range(attn_sample.shape[0]):
                row_str = " ".join([f"{val:.3f}" for val in attn_sample[i, :min(10, attn_sample.shape[1])]])
                print(f"Token {i:2d}: {row_str}")
        else:
            print("Attention Matrix (first 10x10):")
            for i in range(10):
                row_str = " ".join([f"{val:.3f}" for val in attn_sample[i, :10]])
                print(f"Token {i:2d}: {row_str}")
                
def upsample_vit_attention(att_cls_vector, image_shape, vit_grid_shape):
    """
    Upsample ViT attention class token vector to match the image shape.
    Creates a voxel-wise attention volume with the class token attention values.
    
    Args:
        att_cls_vector: Attention class token vector from ViT excluding CLS token (shape: [sequence_length-1])
        image_shape: Target image shape (D_img, H_img, W_img)
        vit_grid_shape: ViT grid shape (D_vit, H_vit, W_vit)
    Returns:
        att_volume: Upsampled attention volume (shape: [D_img, H_img, W_img])
    """

    B, seq_len = att_cls_vector.shape
    
    D_img, H_img, W_img = image_shape
    D_vit, H_vit, W_vit = vit_grid_shape

    assert seq_len == D_vit * H_vit * W_vit, \
        f"att_cls_token_vector length {len(att_cls_vector)} mismatch with vit_grid_shape {vit_grid_shape}" 
    
    att_cls_vector = att_cls_vector.reshape(B, 1, D_vit, H_vit, W_vit)  # Reshape to (B, C, D_vit, H_vit, W_vit)
    att_volume = F.interpolate(
        att_cls_vector,
        size=(D_img, H_img, W_img),
        mode='nearest-exact'
    )
    
    att_volume = att_volume.squeeze(1) # Remove channel dimension, shape: (B, D_img, H_img, W_img)
                
    return att_volume
    
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    resnet_config = {
        'block': 'basic',
        'layers': [2, 2, 2, 2],
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
    'save_attn': False,
    'save_attn_logits': False
    }
    
    purevit_config = {
    'in_channels': 1,
    'img_size': (96,112,96),
    'patch_size': (16,16,16),
    'num_heads': 12,
    'hidden_size': 768,
    'mlp_dim': 3072,
    'num_layers': 12,
    'proj_type': 'perceptron',
    'pos_embed_type': 'sincos',
    'classification': True,
    'num_classes': 1,
    'dropout_rate': 0.0,
    'spatial_dims': 3,
    'post_activation': 'none',
    'qkv_bias': False,
    'save_attn': True,
    'save_attn_logits': False
    }
    
    USE_MODEL = "hybrid" # Choose model: "hybrid", "resnet_extractor", "purevit"
    RETURN_ATTENTION = 'raw' # Choose whether to get raw attention logits or softmaxed attention logits or none at all. Options: 'raw', 'softmaxed', other string.
    
    if RETURN_ATTENTION == 'raw' or RETURN_ATTENTION == 'softmaxed':
        print(f"Saving attention maps: {RETURN_ATTENTION}")
        purevit_config['save_attn'] = True
        vit_config['save_attn'] = True
        purevit_config['save_attn_logits'] = True
        vit_config['save_attn_logits'] = True
    else:
        print("Not saving attention maps")
        purevit_config['save_attn'] = False
        vit_config['save_attn'] = False
        purevit_config['save_attn_logits'] = False
        vit_config['save_attn_logits'] = False


    if USE_MODEL == "resnet_extractor":
        print("Using ResNet Feature Extractor")
        model = ResnetFeatureExtractor(resnet_config=resnet_config).to(device)
        dummy_input = torch.randn(1, 1, 91, 109, 91).to(device)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
    elif USE_MODEL == "hybrid":
        print("Using Hybrid model")
        resnet_extractor = ResnetFeatureExtractor(resnet_config=resnet_config)
        model = ResNetViT(resnet_feature_extractor=resnet_extractor, vit_config=vit_config).to(device)
        
        dummy_input = torch.randn(1, 1, 91, 109, 91).to(device)
        output, attention_maps = model(dummy_input, return_attention=RETURN_ATTENTION)
        print(f"Output shape: {output.shape}")
          
    elif USE_MODEL == "purevit":
        print("Using Pure ViT")
        model = pureViT(vit_config=purevit_config).to(device)
        
        dummy_input = torch.randn(1, 1, 96, 112, 96).to(device)
        output, attention_maps = model(dummy_input, return_attention=RETURN_ATTENTION)
        print(f"Output shape: {output.shape}")
            
    else:
        raise ValueError("Invalid model choice. Use 'resnet_extractor' or 'purevit'")
    
    if (RETURN_ATTENTION == 'raw' or RETURN_ATTENTION == 'softmaxed') and USE_MODEL != "resnet_extractor":
        print(f"Attention maps shape: {len(attention_maps)} layers, each with shape {attention_maps[0].shape}")
        attn_map = model.get_attention_map(layer=7, head=None, average_heads=False)
        print(f"Attention map shape: {attn_map.shape}")
            
        #model.visualize_attention_pattern(layer_idx=-1)
    
    
    
            