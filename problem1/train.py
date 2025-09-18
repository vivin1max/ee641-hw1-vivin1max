import os
import json
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import ShapeDetectionDataset, detection_collate_fn
from .model import MultiScaleDetector
from .loss import DetectionLoss
from .utils import generate_anchors


def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    model.train()
    running = {"loss_total": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_loc": 0.0, "num_pos": 0.0}
    for images, targets in dataloader:
        images = images.to(device)
        preds = model(images)
        loss_dict = criterion(preds, targets, anchors)

        optimizer.zero_grad()
        loss_dict["loss_total"].backward()
        optimizer.step()

        for k in ["loss_total", "loss_obj", "loss_cls", "loss_loc"]:
            running[k] += float(loss_dict[k].item())
        running["num_pos"] += float(loss_dict["num_pos"].item())
    n = len(dataloader)
    for k in running:
        running[k] /= max(n, 1)
    return running


@torch.no_grad()
def validate(model, dataloader, criterion, device, anchors):
    model.eval()
    running = {"loss_total": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_loc": 0.0, "num_pos": 0.0}
    for images, targets in dataloader:
        images = images.to(device)
        preds = model(images)
        loss_dict = criterion(preds, targets, anchors)
        for k in ["loss_total", "loss_obj", "loss_cls", "loss_loc"]:
            running[k] += float(loss_dict[k].item())
        running["num_pos"] += float(loss_dict["num_pos"].item())
    n = len(dataloader)
    for k in running:
        running[k] /= max(n, 1)
    return running


def main():
    # Configuration
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    num_workers = 0  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(root, 'datasets', 'detection')
    train_img_dir = os.path.join(data_root, 'train')
    val_img_dir = os.path.join(data_root, 'val')
    train_ann = os.path.join(data_root, 'train_annotations.json')
    val_ann = os.path.join(data_root, 'val_annotations.json')

    results_dir = os.path.join(root, 'problem1', 'results')
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    log_path = os.path.join(results_dir, 'training_log.json')
    best_model_path = os.path.join(results_dir, 'best_model.pth')

    # Datasets
    train_ds = ShapeDetectionDataset(train_img_dir, train_ann)
    val_ds = ShapeDetectionDataset(val_img_dir, val_ann)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)

    # Model and loss
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # Anchors 
    fm_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(fm_sizes, anchor_scales, image_size=224)
    anchors = [a.to(device) for a in anchors]

    history = {"train": [], "val": []}
    best_val = float('inf')
    for epoch in range(1, num_epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device, anchors)
        va = validate(model, val_loader, criterion, device, anchors)
        history["train"].append(tr)
        history["val"].append(va)

        # Save best model based on val loss_total
        if va["loss_total"] < best_val:
            best_val = va["loss_total"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val,
            }, best_model_path)

        with open(log_path, 'w') as f:
            json.dump({"history": history, "best_val": best_val}, f, indent=2)

        print(f"Epoch {epoch:03d} | train {tr['loss_total']:.4f} | val {va['loss_total']:.4f} | pos {va['num_pos']:.1f}")


if __name__ == '__main__':
    main()
