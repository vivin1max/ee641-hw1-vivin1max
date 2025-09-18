import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
   
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        self.annotations = {}
        for img in data['images']:
            self.annotations[img['file_name']] = img['keypoints']
        self.image_files = sorted(self.annotations.keys())
    
    def generate_heatmap(self, keypoints, height, width):
        num_keypoints = len(keypoints)
        heatmaps = torch.zeros((num_keypoints, height, width))
        
        # Creating coordinate grid
        y = torch.arange(height).float()
        x = torch.arange(width).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        scale_x = width / 128
        scale_y = height / 128
        keypoints = torch.tensor(keypoints).float()  
        keypoints[:, 0] = keypoints[:, 0] * scale_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y

        # Generating gaussian heatmap for each keypoint
        for i in range(num_keypoints):
            x_p, y_p = keypoints[i]
            
            # Gaussian formula:
            gaussian = torch.exp(-((xx - x_p)**2 + (yy - y_p)**2) / (2 * self.sigma**2))
            heatmaps[i] = gaussian
            
        return heatmaps

    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('L')  
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.unsqueeze(0)
        keypoints = torch.tensor(self.annotations[self.image_files[idx]])  
        
        if self.output_type == 'heatmap':
            # Generate heatmaps
            targets = self.generate_heatmap(keypoints, self.heatmap_size, self.heatmap_size)
        else:  # regression    
            targets = keypoints.clone().float()
            targets[:, 0] /= 128.0 
            targets[:, 1] /= 128.0  
            targets = targets.flatten()  
            
        return image, targets

    def __len__(self):
        return len(self.image_files)

    def get_keypoints(self, idx):
        key = self.image_files[idx]
        kps = self.annotations[key]
        return np.array(kps, dtype=np.float32)