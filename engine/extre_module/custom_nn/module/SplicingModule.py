'''
本文件由BiliBili：魔傀面具整理
Splicing Module for feature concatenation and fusion
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv


class SplicingModule(nn.Module):
    """
    Splicing Module for concatenating and fusing features from multiple sources.
    
    This module takes multiple feature maps as input, concatenates them along the channel dimension,
    and then processes them through convolution layers to produce a unified feature representation.
    
    Args:
        inc (list): List of input channel numbers for each feature map
        ouc (int): Output channel number
        reduction (int): Reduction factor for intermediate channels (default: 2)
    """
    def __init__(self, inc, ouc, reduction=2):
        super(SplicingModule, self).__init__()
        
        self.num_inputs = len(inc)
        self.total_channels = sum(inc)
        
        # Channel alignment - convert all inputs to same dimension
        self.channel_align = nn.ModuleList([])
        for in_ch in inc:
            if in_ch != ouc:
                self.channel_align.append(Conv(in_ch, ouc, 1))
            else:
                self.channel_align.append(nn.Identity())
        
        # Intermediate channels after concatenation
        intermediate_ch = max(int(ouc * self.num_inputs / reduction), ouc)
        
        # Fusion convolutions
        self.fusion_conv = nn.Sequential(
            Conv(ouc * self.num_inputs, intermediate_ch, 3),
            Conv(intermediate_ch, ouc, 3)
        )
        
        # Channel attention for feature weighting
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ouc * self.num_inputs, intermediate_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_ch, ouc * self.num_inputs, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x (list): List of feature tensors to be spliced
            
        Returns:
            torch.Tensor: Fused output feature map
        """
        # Align channels
        aligned_feats = []
        for idx, feat in enumerate(x):
            aligned_feats.append(self.channel_align[idx](feat))
        
        # Concatenate features
        concat_feats = torch.cat(aligned_feats, dim=1)
        
        # Apply channel attention
        attn_weights = self.channel_attention(concat_feats)
        weighted_feats = concat_feats * attn_weights
        
        # Fusion
        out = self.fusion_conv(weighted_feats)
        
        return out


class AdaptiveSplicingModule(nn.Module):
    """
    Adaptive Splicing Module with spatial size handling.
    
    This module can handle features with different spatial sizes by upsampling
    smaller features to match the largest spatial size before concatenation.
    
    Args:
        inc (list): List of input channel numbers for each feature map
        ouc (int): Output channel number
        reduction (int): Reduction factor for intermediate channels (default: 2)
        upsample_mode (str): Upsampling mode ('nearest' or 'bilinear', default: 'bilinear')
    """
    def __init__(self, inc, ouc, reduction=2, upsample_mode='bilinear'):
        super(AdaptiveSplicingModule, self).__init__()
        
        self.num_inputs = len(inc)
        self.upsample_mode = upsample_mode
        
        # Channel alignment
        self.channel_align = nn.ModuleList([])
        for in_ch in inc:
            if in_ch != ouc:
                self.channel_align.append(Conv(in_ch, ouc, 1))
            else:
                self.channel_align.append(nn.Identity())
        
        # Intermediate channels
        intermediate_ch = max(int(ouc * self.num_inputs / reduction), ouc)
        
        # Feature weighting
        self.feature_weights = nn.Parameter(torch.ones(self.num_inputs))
        
        # Fusion network
        self.fusion_conv = nn.Sequential(
            Conv(ouc * self.num_inputs, intermediate_ch, 3),
            Conv(intermediate_ch, ouc, 3)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(ouc * self.num_inputs, ouc, 1),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
            nn.Conv2d(ouc, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x (list): List of feature tensors to be spliced (can have different spatial sizes)
            
        Returns:
            torch.Tensor: Fused output feature map
        """
        # Find target spatial size (largest)
        sizes = [feat.shape[2:] for feat in x]
        target_size = tuple(max(s) for s in zip(*sizes))
        
        # Align channels and spatial sizes
        aligned_feats = []
        for idx, feat in enumerate(x):
            # Channel alignment
            feat_aligned = self.channel_align[idx](feat)
            
            # Spatial alignment
            if feat_aligned.shape[2:] != target_size:
                feat_aligned = F.interpolate(
                    feat_aligned, 
                    size=target_size, 
                    mode=self.upsample_mode,
                    align_corners=False if self.upsample_mode == 'bilinear' else None
                )
            
            # Apply learned weights
            feat_aligned = feat_aligned * self.feature_weights[idx]
            aligned_feats.append(feat_aligned)
        
        # Concatenate features
        concat_feats = torch.cat(aligned_feats, dim=1)
        
        # Apply spatial attention
        spatial_attn = self.spatial_attention(concat_feats)
        weighted_feats = concat_feats * spatial_attn
        
        # Fusion
        out = self.fusion_conv(weighted_feats)
        
        return out


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(YELLOW + '='*80 + RESET)
    print(YELLOW + 'Testing SplicingModule' + RESET)
    print(YELLOW + '='*80 + RESET)
    
    # Test 1: Basic SplicingModule with same spatial size
    batch_size, channel_1, channel_2, channel_3, height, width = 1, 32, 64, 48, 32, 32
    ouc_channel = 64
    
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)
    inputs_3 = torch.randn((batch_size, channel_3, height, width)).to(device)
    
    module = SplicingModule([channel_1, channel_2, channel_3], ouc_channel).to(device)
    
    outputs = module([inputs_1, inputs_2, inputs_3])
    print(GREEN + f'Input 1 size: {inputs_1.size()}' + RESET)
    print(GREEN + f'Input 2 size: {inputs_2.size()}' + RESET)
    print(GREEN + f'Input 3 size: {inputs_3.size()}' + RESET)
    print(GREEN + f'Output size: {outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,
        args=[[inputs_1, inputs_2, inputs_3]],
        output_as_string=True,
        output_precision=4,
        print_detailed=True
    )
    print(RESET)
    
    print('\n' + YELLOW + '='*80 + RESET)
    print(YELLOW + 'Testing AdaptiveSplicingModule' + RESET)
    print(YELLOW + '='*80 + RESET)
    
    # Test 2: AdaptiveSplicingModule with different spatial sizes
    batch_size = 1
    channel_1, height_1, width_1 = 32, 64, 64
    channel_2, height_2, width_2 = 64, 32, 32
    channel_3, height_3, width_3 = 48, 16, 16
    ouc_channel = 64
    
    inputs_1 = torch.randn((batch_size, channel_1, height_1, width_1)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height_2, width_2)).to(device)
    inputs_3 = torch.randn((batch_size, channel_3, height_3, width_3)).to(device)
    
    module = AdaptiveSplicingModule([channel_1, channel_2, channel_3], ouc_channel).to(device)
    
    outputs = module([inputs_1, inputs_2, inputs_3])
    print(GREEN + f'Input 1 size: {inputs_1.size()}' + RESET)
    print(GREEN + f'Input 2 size: {inputs_2.size()}' + RESET)
    print(GREEN + f'Input 3 size: {inputs_3.size()}' + RESET)
    print(GREEN + f'Output size: {outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(
        model=module,
        args=[[inputs_1, inputs_2, inputs_3]],
        output_as_string=True,
        output_precision=4,
        print_detailed=True
    )
    print(RESET)
