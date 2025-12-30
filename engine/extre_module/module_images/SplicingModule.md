# Splicing Module

## Overview
The Splicing Module is a feature fusion component designed for concatenating and processing features from multiple sources in deep neural networks. This module is particularly useful in multi-scale feature fusion tasks commonly found in object detection and computer vision applications.

## Features

### 1. SplicingModule
Basic splicing module for concatenating and fusing features from multiple sources with the same spatial dimensions.

**Key Features:**
- Channel alignment for inputs with different channel dimensions
- Channel attention mechanism for adaptive feature weighting
- Fusion convolutions to integrate concatenated features

**Usage:**
```python
from engine.extre_module.custom_nn.module.SplicingModule import SplicingModule

# Define input channels and output channels
inc = [32, 64, 48]  # Three feature maps with different channels
ouc = 64            # Output channel dimension

# Create module
module = SplicingModule(inc, ouc, reduction=2)

# Forward pass
inputs = [feat1, feat2, feat3]  # List of feature tensors
output = module(inputs)
```

### 2. AdaptiveSplicingModule
Advanced splicing module that handles features with different spatial sizes through adaptive upsampling.

**Key Features:**
- Automatic spatial size alignment through interpolation
- Learnable feature weighting parameters
- Spatial attention mechanism for enhanced feature selection
- Support for different upsampling modes (bilinear, nearest)

**Usage:**
```python
from engine.extre_module.custom_nn.module.SplicingModule import AdaptiveSplicingModule

# Define input channels and output channels
inc = [32, 64, 48]  # Three feature maps with different channels
ouc = 64            # Output channel dimension

# Create module with bilinear upsampling
module = AdaptiveSplicingModule(inc, ouc, reduction=2, upsample_mode='bilinear')

# Forward pass with different spatial sizes
inputs = [feat1, feat2, feat3]  # Can have different H, W dimensions
output = module(inputs)
```

## Architecture Details

### SplicingModule Architecture:
1. **Channel Alignment Layer**: 1x1 convolutions to align input channels
2. **Concatenation**: Concatenate aligned features along channel dimension
3. **Channel Attention**: Global average pooling + MLP to compute attention weights
4. **Fusion Convolutions**: 3x3 convolutions to process weighted features

### AdaptiveSplicingModule Architecture:
1. **Channel Alignment Layer**: 1x1 convolutions to align input channels
2. **Spatial Alignment**: Interpolation to match largest spatial size
3. **Learnable Weighting**: Per-input learnable scalar weights
4. **Concatenation**: Concatenate aligned and weighted features
5. **Spatial Attention**: Convolutional layers to compute spatial attention map
6. **Fusion Convolutions**: 3x3 convolutions to process attended features

## Parameters

### Common Parameters:
- `inc` (list): List of input channel numbers for each feature map
- `ouc` (int): Output channel number
- `reduction` (int): Reduction factor for intermediate channels (default: 2)

### AdaptiveSplicingModule Additional Parameters:
- `upsample_mode` (str): Upsampling interpolation mode ('bilinear' or 'nearest', default: 'bilinear')

## Examples

### Example 1: Basic Feature Fusion
```python
import torch
from engine.extre_module.custom_nn.module.SplicingModule import SplicingModule

# Create three feature maps with same spatial size
feat1 = torch.randn(1, 32, 64, 64)   # [B, C, H, W]
feat2 = torch.randn(1, 64, 64, 64)
feat3 = torch.randn(1, 48, 64, 64)

# Create splicing module
module = SplicingModule([32, 64, 48], 64)

# Fuse features
output = module([feat1, feat2, feat3])  # Output: [1, 64, 64, 64]
```

### Example 2: Multi-Scale Feature Fusion
```python
import torch
from engine.extre_module.custom_nn.module.SplicingModule import AdaptiveSplicingModule

# Create three feature maps with different spatial sizes
feat1 = torch.randn(1, 32, 64, 64)   # High resolution
feat2 = torch.randn(1, 64, 32, 32)   # Medium resolution
feat3 = torch.randn(1, 48, 16, 16)   # Low resolution

# Create adaptive splicing module
module = AdaptiveSplicingModule([32, 64, 48], 64)

# Fuse multi-scale features
output = module([feat1, feat2, feat3])  # Output: [1, 64, 64, 64]
```

## Testing

To test the module:
```bash
cd /home/runner/work/DEIM-DEIM/DEIM-DEIM
python engine/extre_module/custom_nn/module/SplicingModule.py
```

Or:
```bash
python engine/extre_module/custom_nn/featurefusion/SplicingModule.py
```

## Integration with DEIM

The Splicing Module can be integrated into DEIM architectures for:
- Multi-scale feature pyramid fusion
- Encoder-decoder feature concatenation
- Cross-stage feature integration
- Attention-based feature selection

## Location in Repository

The module is available in two locations:
1. `/engine/extre_module/custom_nn/module/SplicingModule.py` - General module location
2. `/engine/extre_module/custom_nn/featurefusion/SplicingModule.py` - Feature fusion specific location

Both files contain identical implementations.

## Dependencies

- PyTorch
- engine.extre_module.ultralytics_nn.conv (Conv layer)
- calflops (for FLOPs calculation in testing)

## Notes

- The module automatically handles channel alignment through 1x1 convolutions
- AdaptiveSplicingModule is recommended for multi-scale feature fusion tasks
- The reduction parameter controls the trade-off between model capacity and efficiency
- Both modules use residual-like connections through attention mechanisms for better gradient flow
