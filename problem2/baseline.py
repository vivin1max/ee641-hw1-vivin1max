import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import os
from dataset import KeypointDataset

from model import HeatmapNet, RegressionNet
from evaluate import evaluate_model, compute_pck, plot_pck_curves, extract_keypoints_from_heatmaps
from train import train_heatmap_model


def ablation_study(dataset_params, model_class, device, save_path):
    image_dir = dataset_params['image_dir']
    annotation_file = dataset_params['annotation_file']
    results = {}
    
    # Experiment 1: Heatmap Resolution
    resolutions = [32, 64, 128]
    resolution_results = {}
    
    for res in resolutions:
        print(f"\nTesting heatmap resolution: {res}x{res}")
        train_dataset = KeypointDataset(image_dir, annotation_file, output_type='heatmap', heatmap_size=res)
        val_dataset = KeypointDataset(image_dir, annotation_file, output_type='heatmap', heatmap_size=res)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        model = model_class().to(device)
        # Use new training function
        train_losses, val_losses = train_heatmap_model(model, train_loader, val_loader, num_epochs=10)
        # Evaluate
        test_loader = DataLoader(val_dataset, batch_size=32)
        preds, gts, _ = evaluate_model(model, test_loader, device, is_heatmap=True)
        pck = compute_pck(preds, gts, thresholds=[0.1])
        resolution_results[res] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'pck': pck
        }
    
    results['resolution_study'] = resolution_results
    
    # Experiment 2: Gaussian Sigma
    sigmas = [1.0, 2.0, 3.0, 4.0]
    sigma_results = {}
    
    for sigma in sigmas:
        print(f"\nTesting Gaussian sigma: {sigma}")
        train_dataset = KeypointDataset(image_dir, annotation_file, output_type='heatmap', sigma=sigma)
        val_dataset = KeypointDataset(image_dir, annotation_file, output_type='heatmap', sigma=sigma)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        model = model_class().to(device)
        train_losses, val_losses = train_heatmap_model(model, train_loader, val_loader, num_epochs=10)
        test_loader = DataLoader(val_dataset, batch_size=32)
        preds, gts, _ = evaluate_model(model, test_loader, device, is_heatmap=True)
        pck = compute_pck(preds, gts, thresholds=[0.1])
        sigma_results[sigma] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'pck': pck
        }
    
    results['sigma_study'] = sigma_results
    with open(os.path.join(save_path, 'ablation_study_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    plot_ablation_results(results, save_path)

def analyze_failure_cases(model_heatmap, model_regression, test_loader, device, save_path):

    model_heatmap.eval()
    model_regression.eval()
    failure_threshold = 0.1
    success_threshold = 0.05
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            heatmap_out = model_heatmap(images)
            regression_out = model_regression(images)

            # heatmap_out: 
            heatmap_coords_norm, heatmap_coords_hm_px = extract_keypoints_from_heatmaps(heatmap_out)

            # regression_out: 
            regression_coords_norm = regression_out.view(-1, 5, 2)

            # Map heatmap pixel coords to image pixel coords
            hm_h = heatmap_out.shape[-2]
            hm_w = heatmap_out.shape[-1]
            img_h = images.shape[-2]
            img_w = images.shape[-1]
            scale_x = float(img_w) / float(hm_w)
            scale_y = float(img_h) / float(hm_h)
            heatmap_coords_img_px = heatmap_coords_hm_px.clone()
            heatmap_coords_img_px[:, :, 0] = heatmap_coords_hm_px[:, :, 0] * scale_x
            heatmap_coords_img_px[:, :, 1] = heatmap_coords_hm_px[:, :, 1] * scale_y

            # Regression pixel coords
            regression_coords_img_px = regression_coords_norm.clone()
            regression_coords_img_px[:, :, 0] = regression_coords_norm[:, :, 0] * float(img_w)
            regression_coords_img_px[:, :, 1] = regression_coords_norm[:, :, 1] * float(img_h)
            batch_size = images.size(0)
            gts = []
            for b in range(batch_size):
                sample_idx = batch_idx * test_loader.batch_size + b
                if sample_idx < len(test_loader.dataset):
                    kp_px = test_loader.dataset.get_keypoints(sample_idx)  
                    gts.append(torch.from_numpy(kp_px).float())
                else:
                    gts.append(torch.zeros(5, 2))
            gt_img_px = torch.stack(gts, dim=0) 
            gt_norm = gt_img_px.clone()
            gt_norm[:, :, 0] = gt_img_px[:, :, 0] / float(img_w)
            gt_norm[:, :, 1] = gt_img_px[:, :, 1] / float(img_h)

            heatmap_errors = torch.norm(heatmap_coords_norm - gt_norm, dim=2)
            regression_errors = torch.norm(regression_coords_norm - gt_norm, dim=2)
            for i in range(len(images)):
                heatmap_failed = (heatmap_errors[i].mean() > failure_threshold)
                regression_failed = (regression_errors[i].mean() > failure_threshold)
                
                if heatmap_failed and not regression_failed:
                    # Case 1: Heatmap fails, regression succeeds
                    save_failure_case(images[i], heatmap_coords_img_px[i], regression_coords_img_px[i],
                                   gt_img_px[i], save_path, f'case1_batch{batch_idx}_img{i}')

                elif not heatmap_failed and regression_failed:
                    # Case 2: Regression fails, heatmap succeeds
                    save_failure_case(images[i], heatmap_coords_img_px[i], regression_coords_img_px[i],
                                   gt_img_px[i], save_path, f'case2_batch{batch_idx}_img{i}')

                elif heatmap_failed and regression_failed:
                    # Case 3: Both fail
                    save_failure_case(images[i], heatmap_coords_img_px[i], regression_coords_img_px[i],
                                   gt_img_px[i], save_path, f'case3_batch{batch_idx}_img{i}')

def save_failure_case(image, heatmap_pred_px, regression_pred_px, gt_px, save_path, name):

    import numpy as np
    plt.figure(figsize=(12, 4))
    gt_x = gt_px[:, 0].cpu().numpy()
    gt_y = gt_px[:, 1].cpu().numpy()
    hm_x = heatmap_pred_px[:, 0].cpu().numpy()
    hm_y = heatmap_pred_px[:, 1].cpu().numpy()
    rg_x = regression_pred_px[:, 0].cpu().numpy()
    rg_y = regression_pred_px[:, 1].cpu().numpy()

    def add_offsets(x_coords, y_coords):
        pts = list(zip(x_coords, y_coords))
        seen = {}
        offsets = []
        for i, (px, py) in enumerate(pts):
            key = (round(px, 1), round(py, 1))
            if key in seen:
                seen[key].append(i)
            else:
                seen[key] = [i]

        offsets = [(0.0, 0.0)] * len(pts)
        for _, idxs in seen.items():
            if len(idxs) > 1:
                n = len(idxs)
                for j, idx in enumerate(idxs):
                    angle = 2 * 3.14159 * j / n
                    offsets[idx] = (3.0 * np.cos(angle), 3.0 * np.sin(angle))

        x_off = [x + dx for (x, y), (dx, dy) in zip(pts, offsets)]
        y_off = [y + dy for (x, y), (dx, dy) in zip(pts, offsets)]
        return x_off, y_off

    # Original image with ground truth
    plt.subplot(131)
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    gt_x_off, gt_y_off = add_offsets(gt_x, gt_y)
    for i in range(len(gt_x_off)):
        plt.scatter(gt_x_off[i], gt_y_off[i], c='g', marker='o')
        plt.text(gt_x_off[i] + 2, gt_y_off[i] + 2, f'GT{i}', color='g', fontsize=8)
    plt.title('Ground Truth')

    # Heatmap prediction
    plt.subplot(132)
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    hm_x_off, hm_y_off = add_offsets(hm_x, hm_y)
    for i in range(len(hm_x_off)):
        plt.scatter(hm_x_off[i], hm_y_off[i], c='r', marker='x')
        plt.text(hm_x_off[i] + 2, hm_y_off[i] + 2, f'H{i}', color='r', fontsize=8)
    for i in range(len(gt_x_off)):
        plt.scatter(gt_x_off[i], gt_y_off[i], c='g', marker='o')
        plt.text(gt_x_off[i] + 2, gt_y_off[i] - 8, f'GT{i}', color='g', fontsize=8)
    plt.title('Heatmap Prediction')

    # Regression prediction
    plt.subplot(133)
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    rg_x_off, rg_y_off = add_offsets(rg_x, rg_y)
    for i in range(len(rg_x_off)):
        plt.scatter(rg_x_off[i], rg_y_off[i], c='b', marker='x')
        plt.text(rg_x_off[i] + 2, rg_y_off[i] + 2, f'R{i}', color='b', fontsize=8)
    for i in range(len(gt_x_off)):
        plt.scatter(gt_x_off[i], gt_y_off[i], c='g', marker='o')
        plt.text(gt_x_off[i] + 2, gt_y_off[i] - 8, f'GT{i}', color='g', fontsize=8)
    plt.title('Regression Prediction')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'failure_{name}.png'))
    plt.close()

def plot_ablation_results(results, save_path):
    # Plot resolution study
    plt.figure(figsize=(10, 5))
    for res, data in results['resolution_study'].items():
        plt.plot(data['val_losses'], label=f'Resolution {res}x{res}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Heatmap Resolution')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'resolution_study.png'))
    plt.close()
    
    # Plot sigma study
    plt.figure(figsize=(10, 5))
    for sigma, data in results['sigma_study'].items():
        plt.plot(data['val_losses'], label=f'Sigma {sigma}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Effect of Gaussian Sigma')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'sigma_study.png'))
    plt.close()

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'results/ablation_study'
    os.makedirs(save_path, exist_ok=True)
    
    # Dataset parameters
    dataset_params = {
        'image_dir': '../datasets/keypoints/train',
        'annotation_file': '../datasets/keypoints/train_annotations.json'
    }
    print("Running ablation studies...")
    ablation_study(dataset_params, HeatmapNet, device, save_path)
    
    # Load trained models for failure analysis
    heatmap_model = HeatmapNet().to(device)
    regression_model = RegressionNet().to(device)
    
    heatmap_model.load_state_dict(torch.load('results/heatmap_model.pth'))
    regression_model.load_state_dict(torch.load('results/regression_model.pth'))
    
    # Create test dataset
    test_dataset = KeypointDataset(dataset_params['image_dir'],
                                 dataset_params['annotation_file'],
                                 output_type='heatmap')
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Analyze failure cases
    print("\nAnalyzing failure cases...")
    analyze_failure_cases(heatmap_model, regression_model, test_loader,
                        device, save_path)

if __name__ == '__main__':
    main()