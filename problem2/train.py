

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet


def train_heatmap_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the heatmap-based model.
    Uses MSE loss between predicted and target heatmaps.
    Logs train/val loss and saves best model.
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_train_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if outputs.dim() == 4 and targets.dim() == 4:
                _, _, Oh, Ow = outputs.shape
                _, _, Th, Tw = targets.shape
                if (Oh != Th) or (Ow != Tw):
                    outputs = nn.functional.interpolate(outputs, size=(Th, Tw), mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                if outputs.dim() == 4 and targets.dim() == 4:
                    _, _, Oh, Ow = outputs.shape
                    _, _, Th, Tw = targets.shape
                    if (Oh != Th) or (Ow != Tw):
                        outputs = nn.functional.interpolate(outputs, size=(Th, Tw), mode='bilinear', align_corners=False)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Heatmap] Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "heatmap_model.pth")
    # Save training log
    with open("heatmap_training_log.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
    return train_losses, val_losses

def train_regression_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the direct regression model.
    Uses MSE loss between predicted and target coordinates.
    Logs train/val loss and saves best model.
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Train
        model.train()
        total_train_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Regression] Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "regression_model.pth")
    # Save training log
    with open("regression_training_log.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
    return train_losses, val_losses


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_image_dir = '../datasets/keypoints/train'
    train_annotation_file = '../datasets/keypoints/train_annotations.json'
    val_image_dir = '../datasets/keypoints/val'
    val_annotation_file = '../datasets/keypoints/val_annotations.json'
    batch_size = 32
    num_epochs = 30
    # Create save directories
    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)
    # Train Heatmap Model
    print("Training Heatmap Model...")
    train_dataset = KeypointDataset(train_image_dir, train_annotation_file, output_type='heatmap')
    val_dataset = KeypointDataset(val_image_dir, val_annotation_file, output_type='heatmap')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    heatmap_model = HeatmapNet().to(device)
    train_heatmap_model(heatmap_model, train_loader, val_loader, num_epochs=num_epochs)
    # Train Regression Model
    print("\nTraining Regression Model...")
    train_dataset = KeypointDataset(train_image_dir, train_annotation_file, output_type='regression')
    val_dataset = KeypointDataset(val_image_dir, val_annotation_file, output_type='regression')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    regression_model = RegressionNet().to(device)
    train_regression_model(regression_model, train_loader, val_loader, num_epochs=num_epochs)

if __name__ == '__main__':
    main()