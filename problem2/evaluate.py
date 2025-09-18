import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
import os

def extract_keypoints_from_heatmaps(heatmaps):
    batch_size, num_keypoints, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    soft = torch.softmax(heatmaps_flat, dim=2)
    xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
    grid_x = xs.repeat(H).view(H, W)  
    grid_y = ys.repeat_interleave(W).view(H, W)  

    grid_x = grid_x.view(1, 1, -1)  
    grid_y = grid_y.view(1, 1, -1)
    expected_x = (soft * grid_x).sum(dim=2)  
    expected_y = (soft * grid_y).sum(dim=2)  
    coords_px = torch.stack([expected_x, expected_y], dim=2)  
    coords_norm = coords_px.clone()
    coords_norm[:, :, 0] /= (W - 1)
    coords_norm[:, :, 1] /= (H - 1)
    return coords_norm, coords_px

def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    N = len(predictions)
    num_keypoints = predictions.shape[1]
    pck_values = {t: 0.0 for t in thresholds}
    
    for i in range(N):
        gt = ground_truths[i]
        pred = predictions[i]
        
        # Computing normalization factor
        if normalize_by == 'bbox':
            # Using bounding box diagonal
            bbox_min = torch.min(gt, dim=0)[0]
            bbox_max = torch.max(gt, dim=0)[0]
            norm_factor = torch.norm(bbox_max - bbox_min)
        else:  
            # Using distance between head and mid-hip
            head = gt[0]  
            left_foot = gt[3]  
            right_foot = gt[4]  
            mid_foot = (left_foot + right_foot) / 2
            norm_factor = torch.norm(head - mid_foot)
        
        # Computing distances
        distances = torch.norm(pred - gt, dim=1)
        distances = distances / norm_factor
        
        # Computing PCK for each threshold
        for t in thresholds:
            correct = (distances <= t).float().mean()
            pck_values[t] += correct.item()
    
    # Average over all samples
    for t in thresholds:
        pck_values[t] /= N
        
    return pck_values

def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    plt.figure(figsize=(10, 6))
    
    thresholds = sorted(pck_heatmap.keys())
    plt.plot(thresholds, [pck_heatmap[t] for t in thresholds], 
             'b-', label='Heatmap Method')
    plt.plot(thresholds, [pck_regression[t] for t in thresholds], 
             'r--', label='Regression Method')
    
    plt.xlabel('Normalized Distance Threshold')
    plt.ylabel('PCK')
    plt.title('PCK Curves: Heatmap vs Regression')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_path, 'pck_curves.png'))
    plt.close()

def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path, title, coords_type='normalized'):
    plt.figure(figsize=(8, 8))
    plt.imshow(image.squeeze(), cmap='gray')
    if coords_type == 'pixel':
        gt_x = gt_keypoints[:, 0].cpu().numpy()
        gt_y = gt_keypoints[:, 1].cpu().numpy()
        pred_x = pred_keypoints[:, 0].cpu().numpy()
        pred_y = pred_keypoints[:, 1].cpu().numpy()
    else:
        if coords_type == 'regression128':
            denom_w = 128.0
            denom_h = 128.0
        else:
            denom_w = float(image.shape[-1] - 1)
            denom_h = float(image.shape[-2] - 1)

        gt_x = (gt_keypoints[:, 0] * denom_w).cpu().numpy()
        gt_y = (gt_keypoints[:, 1] * denom_h).cpu().numpy()
        pred_x = (pred_keypoints[:, 0] * denom_w).cpu().numpy()
        pred_y = (pred_keypoints[:, 1] * denom_h).cpu().numpy()

    # Plotting ground truth points 
    plt.scatter(gt_x, gt_y, c='g', marker='o', label='Ground Truth')
    for i in range(len(gt_x)):
        plt.text(gt_x[i] + 2, gt_y[i] + 2, f'GT{i}', color='g', fontsize=8)

    # Detecting overlapping predicted points
    pts = list(zip(pred_x, pred_y))
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

    # Plotting predicted points with offsets and annotating indices
    pxs = []
    pys = []
    for i, (px, py) in enumerate(pts):
        dx, dy = offsets[i]
        px_off = px + dx
        py_off = py + dy
        pxs.append(px_off)
        pys.append(py_off)
        plt.scatter(px_off, py_off, c='r', marker='x')
        plt.text(px_off + 2, py_off + 2, f'P{i}', color='r', fontsize=8)

    # Drawing lines between corresponding points
    for i in range(len(gt_x)):
        plt.plot([gt_x[i], pxs[i]], [gt_y[i], pys[i]], 'y--', alpha=0.5)

    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, loader, device, is_heatmap=True):
    model.eval()
    all_preds = []
    all_gts = []
    all_images = []
    
    idx_counter = 0
    with torch.no_grad():
        for images, targets in loader:
            batch_size = images.size(0)
            images = images.to(device)
            outputs = model(images)

            img_h = images.shape[-2]
            img_w = images.shape[-1]

            if is_heatmap:
                coords_norm, coords_hm_px = extract_keypoints_from_heatmaps(outputs)
                hm_w = outputs.shape[-1]
                hm_h = outputs.shape[-2]
                scale_x = float(img_w) / float(hm_w)
                scale_y = float(img_h) / float(hm_h)
                coords_img_px = coords_hm_px.clone()
                coords_img_px[:, :, 0] = coords_hm_px[:, :, 0] * scale_x
                coords_img_px[:, :, 1] = coords_hm_px[:, :, 1] * scale_y
                all_preds.append(coords_img_px.cpu())
                gts = []
                for b in range(batch_size):
                    kp_px = loader.dataset.get_keypoints(idx_counter + b)  
                    gts.append(torch.from_numpy(kp_px).float())
                gts = torch.stack(gts, dim=0)  
                all_gts.append(gts)

            else:
                coords = outputs.view(-1, 5, 2)
                coords_img_px = coords * float(img_w)
                all_preds.append(coords_img_px.cpu())
                gts = targets.view(-1, 5, 2) * float(img_w)
                all_gts.append(gts.cpu())
            all_images.append(images.cpu())
            idx_counter += batch_size
    
    return torch.cat(all_preds), torch.cat(all_gts), torch.cat(all_images)

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'results/visualizations'
    os.makedirs(save_path, exist_ok=True)
    
    # Dataset
    test_image_dir = '../datasets/keypoints/val'  
    test_annotation_file = '../datasets/keypoints/val_annotations.json'
    
    # Load models
    heatmap_model = HeatmapNet().to(device)
    regression_model = RegressionNet().to(device)
    
    heatmap_model.load_state_dict(torch.load('results/heatmap_model.pth'))
    regression_model.load_state_dict(torch.load('results/regression_model.pth'))
    
    # Create test dataset and loader
    test_dataset_heatmap = KeypointDataset(test_image_dir, test_annotation_file, 
                                         output_type='heatmap')
    test_dataset_regression = KeypointDataset(test_image_dir, test_annotation_file, 
                                            output_type='regression')
    
    test_loader_heatmap = DataLoader(test_dataset_heatmap, batch_size=32)
    test_loader_regression = DataLoader(test_dataset_regression, batch_size=32)
    
    # Evaluate both models
    heatmap_preds, heatmap_gts, heatmap_images = evaluate_model(
        heatmap_model, test_loader_heatmap, device, is_heatmap=True)
    
    regression_preds, regression_gts, regression_images = evaluate_model(
        regression_model, test_loader_regression, device, is_heatmap=False)
    
    # Compute PCK
    thresholds = [0.05, 0.1, 0.15, 0.2]
    pck_heatmap = compute_pck(heatmap_preds, heatmap_gts, thresholds)
    pck_regression = compute_pck(regression_preds, regression_gts, thresholds)
    
    # Plot PCK curves
    plot_pck_curves(pck_heatmap, pck_regression, save_path)
    
    # Save PCK values
    results = {
        'heatmap_pck': pck_heatmap,
        'regression_pck': pck_regression
    }
    
    import json
    with open(os.path.join('results', 'pck_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Visualize some predictions
    for i in range(min(5, len(heatmap_images))):
        visualize_predictions(
            heatmap_images[i],
            heatmap_preds[i],
            heatmap_gts[i],
            os.path.join(save_path, f'heatmap_pred_{i}.png'),
            'Heatmap Method',
            coords_type='pixel'
        )
        
        visualize_predictions(
            regression_images[i],
            regression_preds[i],
            regression_gts[i],
            os.path.join(save_path, f'regression_pred_{i}.png'),
            'Regression Method',
            coords_type='pixel'
        )

    # Failure case analysis: 
    failures_dir = os.path.join(save_path, 'failures')
    os.makedirs(failures_dir, exist_ok=True)
    # threshold scaled to heatmap normalization 
    img_diag = np.sqrt(128**2 + 128**2)
    thr = 0.1  
    for idx in range(heatmap_preds.shape[0]):
        dists_h = torch.norm(heatmap_preds[idx] - heatmap_gts[idx], dim=1)
        dists_r = torch.norm(regression_preds[idx] - regression_gts[idx], dim=1)
        if dists_h.mean() > thr * img_diag or dists_r.mean() > thr * img_diag:
            img = heatmap_images[idx].unsqueeze(0).to(device)
            with torch.no_grad():
                out = heatmap_model(img)
            out = out.cpu().numpy()[0]  # [K,H,W]
            sample_dir = os.path.join(failures_dir, f'sample_{idx}')
            os.makedirs(sample_dir, exist_ok=True)
            for k in range(out.shape[0]):
                arr = out[k]
                arrn = arr - arr.min()
                if arrn.max() > 0:
                    arrn = (255.0 * arrn / arrn.max()).astype(np.uint8)
                else:
                    arrn = (arrn * 0).astype(np.uint8)
                from PIL import Image
                Image.fromarray(arrn).save(os.path.join(sample_dir, f'kp_{k}.png'))

if __name__ == '__main__':
    main()