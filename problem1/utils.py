from typing import List, Tuple
import torch
import numpy as np


def generate_anchors(feature_map_sizes: List[Tuple[int, int]], anchor_scales: List[List[float]], image_size: int = 224) -> List[torch.Tensor]:

    anchors_all: List[torch.Tensor] = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        stride_h = image_size / H
        stride_w = image_size / W
        centers_y = (np.arange(H) + 0.5) * stride_h
        centers_x = (np.arange(W) + 0.5) * stride_w
        cy, cx = np.meshgrid(centers_y, centers_x, indexing='ij')  
        cx = cx.reshape(-1)
        cy = cy.reshape(-1)
        anchors = []
        for s in scales:
            half = s / 2.0
            x1 = cx - half
            y1 = cy - half
            x2 = cx + half
            y2 = cy + half
            boxes = np.stack([x1, y1, x2, y2], axis=1) 
            anchors.append(boxes)
        anchors = np.concatenate(anchors, axis=0)  
        anchors_all.append(torch.from_numpy(anchors).float())
    return anchors_all


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    b1 = boxes1.unsqueeze(1)  
    b2 = boxes2.unsqueeze(0)  
    x1 = torch.maximum(b1[..., 0], b2[..., 0])
    y1 = torch.maximum(b1[..., 1], b2[..., 1])
    x2 = torch.minimum(b1[..., 2], b2[..., 2])
    y2 = torch.minimum(b1[..., 3], b2[..., 3])
    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def encode_boxes(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    # centers and sizes
    ga = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gb = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-6)
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-6)

    aa = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ab = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ah = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)

    tx = (ga - aa) / aw
    ty = (gb - ab) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    aa = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ab = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = (anchors[:, 2] - anchors[:, 0])
    ah = (anchors[:, 3] - anchors[:, 1])

    dx, dy, dw, dh = deltas.unbind(dim=1)
    gx = dx * aw + aa
    gy = dy * ah + ab
    gw = aw * torch.exp(dw)
    gh = ah * torch.exp(dh)
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    return torch.stack([x1, y1, x2, y2], dim=1)


def match_anchors_to_targets(anchors: torch.Tensor, target_boxes: torch.Tensor, target_labels: torch.Tensor,
                             pos_threshold: float = 0.5, neg_threshold: float = 0.3):
    num_anchors = anchors.shape[0]
    matched_labels = torch.zeros((num_anchors,), dtype=torch.long)
    matched_boxes = torch.zeros((num_anchors, 4), dtype=torch.float32)
    pos_mask = torch.zeros((num_anchors,), dtype=torch.bool)
    neg_mask = torch.zeros((num_anchors,), dtype=torch.bool)

    if target_boxes.numel() == 0:
        neg_mask[:] = True
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou = compute_iou(anchors, target_boxes)  
    max_iou, max_idx = iou.max(dim=1)  

    # Positive anchors
    pos_mask = max_iou >= pos_threshold
    matched_boxes[pos_mask] = target_boxes[max_idx[pos_mask]]
    matched_labels[pos_mask] = target_labels[max_idx[pos_mask]] + 1  

    # Negative anchors
    neg_mask = max_iou < neg_threshold

    gt_best_iou, gt_best_anchor = iou.max(dim=0)  # [G]
    force_pos = gt_best_anchor
    pos_mask[force_pos] = True
    matched_boxes[force_pos] = target_boxes
    matched_labels[force_pos] = target_labels + 1
    neg_mask[force_pos] = False

    return matched_labels, matched_boxes, pos_mask, neg_mask


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5, topk: int = 200) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    order = scores.sort(descending=True).indices
    order = order[:topk]
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = compute_iou(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        mask = ious <= iou_thresh
        order = rest[mask]
    return torch.tensor(keep, dtype=torch.long)
