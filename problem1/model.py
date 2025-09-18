import torch
import torch.nn as nn
from typing import List


def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DetectionHead(nn.Module):
    def __init__(self, in_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        out_ch = num_anchors * (5 + num_classes)
        self.pred = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pred(x)
        return x


class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes: int = 3, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone: 4 conv blocks 
        # Block 1 (Stem): 3->32 (s=1), then 32->64 (s=2)
        self.stem1 = conv_bn_relu(3, 32, k=3, s=1, p=1)
        self.stem2 = conv_bn_relu(32, 64, k=3, s=2, p=1)  

        # Block 2: 64->128 (s=2) -> scale1 (112 -> 56)
        self.block2 = conv_bn_relu(64, 128, k=3, s=2, p=1)

        # Block 3: 128->256 (s=2) -> scale2 (56 -> 28)
        self.block3 = conv_bn_relu(128, 256, k=3, s=2, p=1)

        # Block 4: 256->512 (s=2) -> scale3 (28 -> 14)
        self.block4 = conv_bn_relu(256, 512, k=3, s=2, p=1)

        # Detection heads for each scale
        self.head1 = DetectionHead(128, num_anchors, num_classes)
        self.head2 = DetectionHead(256, num_anchors, num_classes)
        self.head3 = DetectionHead(512, num_anchors, num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Backbone forward
        x = self.stem1(x)   
        x = self.stem2(x)   
        f1 = self.block2(x) 
        f2 = self.block3(f1)  
        f3 = self.block4(f2)  

        p1 = self.head1(f1)  
        p2 = self.head2(f2)  
        p3 = self.head3(f3)  
        return [p1, p2, p3]
