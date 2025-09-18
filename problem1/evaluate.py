import os
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .dataset import ShapeDetectionDataset, detection_collate_fn
from .model import MultiScaleDetector
from .utils import generate_anchors, decode_boxes, nms, compute_iou

# Visualization and filtering configs
CLASS_NAMES = ['circle', 'square', 'triangle']
CLASS_COLORS = {
    0: (0, 128, 255),   # circle -> blue
    1: (0, 200, 0),     # square -> green
    2: (255, 0, 0),     # triangle -> red
}

# Filtering thresholds 
OBJ_THRESH = 0.15
COMBINED_THRESH = [0.15, 0.15, 0.20]  
CLS_THRESH = [0.10, 0.10, 0.15]

# Geometry filters 
MIN_BOX_SIZE = 8
ASPECT_RATIO_MIN = 0.75
ASPECT_RATIO_MAX = 1.33

# NMS and caps
NMS_IOU = 0.4
PRE_NMS_TOPK_PER_CLASS_PER_SCALE = 80
TOPK_PER_CLASS = 5
TOPK_TOTAL = 8


@torch.no_grad()
def compute_ap(pred_boxes: List[np.ndarray], pred_scores: List[np.ndarray], gt_boxes: List[np.ndarray], iou_threshold: float = 0.5) -> float:
    all_boxes = np.concatenate(pred_boxes, axis=0) if len(pred_boxes) else np.zeros((0, 4))
    all_scores = np.concatenate(pred_scores, axis=0) if len(pred_scores) else np.zeros((0,))
    img_ids = []
    offset = 0
    for i, pb in enumerate(pred_boxes):
        if pb.size == 0:
            continue
        img_ids.append(np.full((pb.shape[0],), i, dtype=np.int32))
        offset += pb.shape[0]
    img_ids = np.concatenate(img_ids, axis=0) if len(img_ids) else np.zeros((0,), dtype=np.int32)

    # Sort predictions by score desc
    order = np.argsort(-all_scores)
    all_boxes = all_boxes[order]
    img_ids = img_ids[order]

    gt_used = [np.zeros((g.shape[0],), dtype=bool) for g in gt_boxes]

    tp = np.zeros((all_boxes.shape[0],), dtype=np.float32)
    fp = np.zeros((all_boxes.shape[0],), dtype=np.float32)
    for k in range(all_boxes.shape[0]):
        i = img_ids[k]
        if gt_boxes[i].size == 0:
            fp[k] = 1.0
            continue
        ious = iou_np(all_boxes[k:k+1], gt_boxes[i])  # [1,G]
        best = np.argmax(ious[0])
        if ious[0, best] >= iou_threshold and not gt_used[i][best]:
            tp[k] = 1.0
            gt_used[i][best] = True
        else:
            fp[k] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(sum(g.shape[0] for g in gt_boxes), 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = 0.0
        if np.any(recalls >= t):
            p = np.max(precisions[recalls >= t])
        ap += p
    ap /= 11.0
    return float(ap)


def iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], a[:, 2, None], a[:, 3, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    x1 = np.maximum(ax1, bx1)
    y1 = np.maximum(ay1, by1)
    x2 = np.minimum(ax2, bx2)
    y2 = np.minimum(ay2, by2)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-12, None)


def visualize_detections_pil(img: Image.Image, preds: List[Tuple[np.ndarray, int, float]], gts: np.ndarray, save_path: str):
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for box in gts:
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)

    for box, cls, score in preds:
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        color = CLASS_COLORS.get(cls, (255, 165, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{CLASS_NAMES[cls]} {score:.2f}"
        tw = int(draw.textlength(label))
        y_top = max(0, y1 - 12)
        y_bot = min(H - 1, y_top + 12)
        x_right = min(W - 1, x1 + tw + 4)
        draw.rectangle([x1, y_top, x_right, y_bot], fill=color)
        draw.text((x1 + 2, y_top), label, fill=(255, 255, 255))
    img.save(save_path)


@torch.no_grad()
def analyze_scale_performance(model: MultiScaleDetector, dataloader, anchors: List[torch.Tensor], save_dir: str):
    device = next(model.parameters()).device
    size_bins = []
    best_scale_per_gt = []

    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)
        for b in range(images.shape[0]):
            t = targets[b]
            gt = t['boxes'].cpu().numpy()
            if gt.shape[0] == 0:
                continue
            # collect max IoU per GT among scales
            per_gt_best_scale = np.zeros((gt.shape[0],), dtype=np.int32)
            per_gt_best_iou = np.zeros((gt.shape[0],), dtype=np.float32)

            for s, pred in enumerate(outputs):
                B, C, H, W = pred.shape
                a_per_loc = C // (5 + 3)
                pred_b = pred[b].permute(1, 2, 0).contiguous().view(H * W * a_per_loc, 5 + 3)
                deltas = pred_b[:, :4]
                obj = torch.sigmoid(pred_b[:, 4])
                boxes = decode_boxes(deltas, anchors[s].to(device)).cpu()
                # take top-k by obj 
                topk = min(200, boxes.shape[0])
                idx = torch.topk(obj, k=topk).indices.cpu().numpy()
                boxes = boxes.numpy()[idx]
                # IoU to GT
                ious = iou_np(boxes, gt)
                # update per GT best
                for g in range(gt.shape[0]):
                    if ious.shape[0] == 0:
                        continue
                    best_iou = ious[:, g].max()
                    if best_iou > per_gt_best_iou[g]:
                        per_gt_best_iou[g] = best_iou
                        per_gt_best_scale[g] = s

            # sizes from GT
            sizes = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
            size_bins.extend(list(sizes))
            best_scale_per_gt.extend(list(per_gt_best_scale))

    if len(size_bins) == 0:
        return

    size_bins = np.array(size_bins)
    best_scale_per_gt = np.array(best_scale_per_gt)

    # Plot histogram by scale
    plt.figure(figsize=(8, 4))
    bins = np.linspace(0, 200, 21)
    for s in range(3):
        plt.hist(size_bins[best_scale_per_gt == s], bins=bins, alpha=0.5, label=f'Scale {s}')
    plt.xlabel('GT sqrt(area) [px]')
    plt.ylabel('Count')
    plt.title('Scale specialization by GT size')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scale_specialization_hist.png'))
    plt.close()


def run_full_evaluation(results_dir: str):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(root, 'datasets', 'detection')
    val_img_dir = os.path.join(data_root, 'val')
    val_ann = os.path.join(data_root, 'val_annotations.json')

    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    best_model_path = os.path.join(results_dir, 'best_model.pth')

    # Data
    val_ds = ShapeDetectionDataset(val_img_dir, val_ann)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=detection_collate_fn)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Anchors
    fm_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(fm_sizes, anchor_scales, image_size=224)

    # Generate detection results on 10 val images
    dets_per_scale = [0, 0, 0]
    det_count = 0
    for images, targets in val_loader:
        B = images.shape[0]
        images = images.to(device)
        outputs = model(images)
        for b in range(B):
            img_idx = det_count
            per_class_boxes = {c: [] for c in range(3)}
            per_class_scores = {c: [] for c in range(3)}
            for s, pred in enumerate(outputs):
                C = pred.shape[1]
                H, W = pred.shape[2], pred.shape[3]
                a_per_loc = C // (5 + 3)
                pred_b = pred[b].permute(1, 2, 0).contiguous().view(H * W * a_per_loc, 5 + 3)
                deltas = pred_b[:, :4]
                obj = torch.sigmoid(pred_b[:, 4])
                cls_logits = pred_b[:, 5:]
                probs = torch.softmax(cls_logits, dim=1)  
                boxes = decode_boxes(deltas, anchors[s].to(device)).cpu()
                obj = obj.cpu()
                probs = probs.cpu()

                # Geometry masks
                w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
                h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
                ar = torch.where(h > 0, w / torch.clamp(h, min=1e-6), torch.zeros_like(w))
                geom_mask = (w >= MIN_BOX_SIZE) & (h >= MIN_BOX_SIZE) & (ar >= ASPECT_RATIO_MIN) & (ar <= ASPECT_RATIO_MAX)

                # Count detections per scale 
                scale_mask = torch.zeros(boxes.shape[0], dtype=torch.bool)
                for c in range(3):
                    combined = obj * probs[:, c]
                    mask = (obj >= OBJ_THRESH) & (probs[:, c] >= CLS_THRESH[c]) & (combined >= COMBINED_THRESH[c]) & geom_mask
                    scale_mask |= mask
                scale_count = int(scale_mask.sum().item())
                dets_per_scale[s] += scale_count

                for c in range(3):
                    combined = obj * probs[:, c]
                    mask = (obj >= OBJ_THRESH) & (probs[:, c] >= CLS_THRESH[c]) & (combined >= COMBINED_THRESH[c]) & geom_mask
                    if mask.any():
                        idx = torch.topk(combined[mask], k=min(PRE_NMS_TOPK_PER_CLASS_PER_SCALE, mask.sum().item())).indices
                        sel_boxes = boxes[mask][idx]
                        sel_scores = combined[mask][idx]
                        per_class_boxes[c].append(sel_boxes)
                        per_class_scores[c].append(sel_scores)

            final_boxes = []
            final_scores = []
            final_labels = []
            for c in range(3):
                if len(per_class_boxes[c]) == 0:
                    continue
                boxes_c = torch.cat(per_class_boxes[c], dim=0)
                scores_c = torch.cat(per_class_scores[c], dim=0)
                # NMS per class
                keep = nms(boxes_c, scores_c, iou_thresh=NMS_IOU, topk=1000)
                # Keep top-k per class after NMS
                if keep.numel() > 0:
                    boxes_c = boxes_c[keep]
                    scores_c = scores_c[keep]
                    k = min(TOPK_PER_CLASS, boxes_c.shape[0])
                    top_idx = torch.topk(scores_c, k=k).indices
                    final_boxes.append(boxes_c[top_idx])
                    final_scores.append(scores_c[top_idx])
                    final_labels.append(torch.full((k,), c, dtype=torch.long))

            if len(final_boxes) == 0:
                det_count += 1
                if det_count >= 10:
                    break
                continue

            boxes_all = torch.cat(final_boxes, dim=0)
            scores_all = torch.cat(final_scores, dim=0)
            labels_all = torch.cat(final_labels, dim=0)
            k_total = min(TOPK_TOTAL, boxes_all.shape[0])
            idx = torch.topk(scores_all, k=k_total).indices
            boxes_all = boxes_all[idx]
            scores_all = scores_all[idx]
            labels_all = labels_all[idx]

            # Save visualization 
            img_tensor = images[b].detach().cpu().clamp(0, 1)
            np_img = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(np_img)
            gts = targets[b]['boxes'].numpy()
            preds = [(boxes_all[i].detach().numpy(), int(labels_all[i].item()), float(scores_all[i].item())) for i in range(boxes_all.shape[0])]
            save_path = os.path.join(vis_dir, f'detections_{img_idx:02d}.png')
            visualize_detections_pil(pil_img, preds, gts, save_path)

            det_count += 1
            if det_count >= 10:
                break
        if det_count >= 10:
            break

    # Anchor coverage visualization for each scale
    for si, (H, W) in enumerate(fm_sizes):
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        ancs = anchors[si].numpy()
        stride = max(1, ancs.shape[0] // 200)
        for k in range(0, ancs.shape[0], stride):
            x1, y1, x2, y2 = [int(v) for v in ancs[k].tolist()]
            draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)
        img.save(os.path.join(vis_dir, f'anchor_coverage_scale{si}.png'))

    # Scale performance analysis plot
    analyze_scale_performance(model, val_loader, [a.to(device) for a in anchors], vis_dir)

    # Detections-per-scale bar chart
    plt.figure(figsize=(6, 4))
    plt.bar([0, 1, 2], dets_per_scale, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Scale index')
    plt.ylabel('Number of kept detections')
    plt.title('Detections per scale (pre-NMS, above thresh)')
    plt.tight_layout()
    dets_plot_path = os.path.join(vis_dir, 'detections_per_scale.png')
    plt.savefig(dets_plot_path)
    plt.close()

    print(f"Saved visualizations to: {vis_dir}")

    log_path = os.path.join(results_dir, 'training_log.json')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            json.dump({"history": {"train": [], "val": []}, "best_val": None, "note": "Training not run yet. Use python -m problem1.train."}, f, indent=2)

    if not os.path.exists(best_model_path):
        torch.save({'epoch': 0, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': None, 'val_loss': None, 'note': 'Random-initialized weights; train to update.'}, best_model_path)


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(root, 'problem1', 'results')
    os.makedirs(results_dir, exist_ok=True)
    run_full_evaluation(results_dir)
