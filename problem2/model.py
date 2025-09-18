import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, 
                                       stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Encoder (downsampling path)
        self.conv1 = ConvBlock(1, 32)      
        self.conv2 = ConvBlock(32, 64)     
        self.conv3 = ConvBlock(64, 128)    
        self.conv4 = ConvBlock(128, 256)  
        
        # Decoder (upsampling path)
        self.deconv4 = DeconvBlock(256, 128)  
        self.deconv3 = DeconvBlock(256, 64)   
        self.deconv2 = DeconvBlock(128, 32)   
        
        # Final layer
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)      
        x2 = self.conv2(x1)     
        x3 = self.conv3(x2)     
        x4 = self.conv4(x3)
        
        # Decoder with skip connections
        x = self.deconv4(x4)
        x = torch.cat([x, x3], dim=1)  
        x = self.deconv3(x)
        x = torch.cat([x, x2], dim=1)  
        x = self.deconv2(x)
        
        # Final layer 
        x = self.final(x)        
        return x

class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # same encoder architecture
        self.conv1 = ConvBlock(1, 32)      
        self.conv2 = ConvBlock(32, 64)     
        self.conv3 = ConvBlock(64, 128)    
        self.conv4 = ConvBlock(128, 256)   
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_keypoints * 2)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final layer with sigmoid
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x