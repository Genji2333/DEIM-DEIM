"""
Integration Example for Splicing Module

This file demonstrates how to integrate the Splicing Module
into DEIM architectures for multi-scale feature fusion.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import torch.nn as nn

# Import the Splicing Module
from engine.extre_module.custom_nn.module.SplicingModule import (
    SplicingModule, 
    AdaptiveSplicingModule
)


class FeaturePyramidWithSplicing(nn.Module):
    """
    Example Feature Pyramid Network with Splicing Module for fusion
    """
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidWithSplicing, self).__init__()
        
        # Use AdaptiveSplicingModule for multi-scale fusion
        self.splicing = AdaptiveSplicingModule(
            inc=in_channels_list, 
            ouc=out_channels,
            reduction=2,
            upsample_mode='bilinear'
        )
        
    def forward(self, features):
        """
        Args:
            features (list): Multi-scale feature maps [P3, P4, P5]
        
        Returns:
            torch.Tensor: Fused feature map
        """
        fused_features = self.splicing(features)
        return fused_features


class EncoderDecoderWithSplicing(nn.Module):
    """
    Example Encoder-Decoder architecture with Splicing Module
    """
    def __init__(self, encoder_channels, decoder_channels):
        super(EncoderDecoderWithSplicing, self).__init__()
        
        # Use basic SplicingModule for same-size feature fusion
        self.splicing = SplicingModule(
            inc=[encoder_channels, decoder_channels], 
            ouc=decoder_channels,
            reduction=2
        )
        
    def forward(self, encoder_feat, decoder_feat):
        """
        Args:
            encoder_feat: Feature from encoder
            decoder_feat: Feature from decoder
        
        Returns:
            torch.Tensor: Fused feature map
        """
        fused = self.splicing([encoder_feat, decoder_feat])
        return fused


def test_feature_pyramid_splicing():
    """Test Feature Pyramid with Splicing"""
    print("Testing Feature Pyramid with Splicing Module")
    print("-" * 60)
    
    device = torch.device('cpu')
    
    # Simulate multi-scale features (P3, P4, P5)
    p3 = torch.randn(2, 256, 80, 80).to(device)   # High resolution
    p4 = torch.randn(2, 512, 40, 40).to(device)   # Medium resolution
    p5 = torch.randn(2, 1024, 20, 20).to(device)  # Low resolution
    
    # Create FPN with splicing
    fpn = FeaturePyramidWithSplicing(
        in_channels_list=[256, 512, 1024],
        out_channels=256
    ).to(device)
    
    # Forward pass
    fused = fpn([p3, p4, p5])
    
    print(f"P3 shape: {p3.shape}")
    print(f"P4 shape: {p4.shape}")
    print(f"P5 shape: {p5.shape}")
    print(f"Fused output shape: {fused.shape}")
    print(f"Expected output shape: [2, 256, 80, 80]")
    print(f"Test {'PASSED' if fused.shape == torch.Size([2, 256, 80, 80]) else 'FAILED'}")
    print()


def test_encoder_decoder_splicing():
    """Test Encoder-Decoder with Splicing"""
    print("Testing Encoder-Decoder with Splicing Module")
    print("-" * 60)
    
    device = torch.device('cpu')
    
    # Simulate encoder and decoder features with same spatial size
    encoder_feat = torch.randn(2, 512, 32, 32).to(device)
    decoder_feat = torch.randn(2, 256, 32, 32).to(device)
    
    # Create model with splicing
    model = EncoderDecoderWithSplicing(
        encoder_channels=512,
        decoder_channels=256
    ).to(device)
    
    # Forward pass
    fused = model(encoder_feat, decoder_feat)
    
    print(f"Encoder feature shape: {encoder_feat.shape}")
    print(f"Decoder feature shape: {decoder_feat.shape}")
    print(f"Fused output shape: {fused.shape}")
    print(f"Expected output shape: [2, 256, 32, 32]")
    print(f"Test {'PASSED' if fused.shape == torch.Size([2, 256, 32, 32]) else 'FAILED'}")
    print()


def test_basic_splicing():
    """Test basic SplicingModule"""
    print("Testing Basic SplicingModule")
    print("-" * 60)
    
    device = torch.device('cpu')
    
    # Create features with different channels but same spatial size
    feat1 = torch.randn(1, 64, 32, 32).to(device)
    feat2 = torch.randn(1, 128, 32, 32).to(device)
    feat3 = torch.randn(1, 256, 32, 32).to(device)
    
    # Create splicing module
    splicing = SplicingModule(
        inc=[64, 128, 256],
        ouc=128,
        reduction=2
    ).to(device)
    
    # Forward pass
    output = splicing([feat1, feat2, feat3])
    
    print(f"Input 1 shape: {feat1.shape}")
    print(f"Input 2 shape: {feat2.shape}")
    print(f"Input 3 shape: {feat3.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [1, 128, 32, 32]")
    print(f"Test {'PASSED' if output.shape == torch.Size([1, 128, 32, 32]) else 'FAILED'}")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Splicing Module Integration Examples")
    print("=" * 60 + "\n")
    
    test_basic_splicing()
    test_feature_pyramid_splicing()
    test_encoder_decoder_splicing()
    
    print("=" * 60)
    print("All integration tests completed!")
    print("=" * 60)
