from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import match_anchors_to_targets, encode_boxes


class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]], anchors: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        total_obj = 0.0
        total_cls = 0.0
        total_loc = 0.0
        total_pos = 0
        total_neg = 0

        B = len(targets)
        num_scales = len(predictions)

        for s in range(num_scales):
            pred = predictions[s]  
            A = anchors[s].shape[0]  
            _, ch, H, W = pred.shape
            a_per_loc = ch // (5 + self.num_classes)
            assert a_per_loc > 0, "Invalid head channels"

            pred = pred.permute(0, 2, 3, 1).contiguous()  
            pred = pred.view(B, H, W, a_per_loc, 5 + self.num_classes)
            pred = pred.view(B, H * W * a_per_loc, 5 + self.num_classes)

            # Anchors for this scale
            anc = anchors[s].to(pred.device)  

            obj_logits = pred[..., 4]  
            cls_logits = pred[..., 5:]  
            box_deltas = pred[..., :4]  

            for b in range(B):
                t = targets[b]
                t_boxes = t["boxes"].to(pred.device)
                t_labels = t["labels"].to(pred.device)

                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anc, t_boxes, t_labels, pos_threshold=0.5, neg_threshold=0.3
                )

                # Objectness target: 
                obj_target = torch.zeros_like(obj_logits[b])
                obj_target[pos_mask] = 1.0

                # Hard negative mining based on objectness loss
                obj_loss_all = self.bce(obj_logits[b], obj_target)  # [N]
                selected_neg_mask = self.hard_negative_mining(obj_loss_all.detach(), pos_mask, neg_mask, ratio=3)

                # Objectness loss on positives + selected negatives
                obj_mask = pos_mask | selected_neg_mask
                loss_obj = obj_loss_all[obj_mask].sum()

                # Classification loss for positives only
                if pos_mask.any():
                    cls_target = matched_labels[pos_mask] - 1  
                    cls_loss = self.ce(cls_logits[b][pos_mask], cls_target).sum()

                    # Localization loss
                    gt_enc = encode_boxes(matched_boxes[pos_mask], anc[pos_mask])
                    loc_loss = self.smooth_l1(box_deltas[b][pos_mask], gt_enc).sum()
                else:
                    cls_loss = torch.tensor(0.0, device=pred.device)
                    loc_loss = torch.tensor(0.0, device=pred.device)

                total_obj += loss_obj
                total_cls += cls_loss
                total_loc += loc_loss
                total_pos += int(pos_mask.sum().item())
                total_neg += int(selected_neg_mask.sum().item())

        normalizer = max(total_pos, 1)
        loss_obj = total_obj / normalizer
        loss_cls = total_cls / normalizer
        loss_loc = total_loc / normalizer
        loss_total = loss_obj + loss_cls + 2.0 * loss_loc

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_total": loss_total,
            "num_pos": torch.tensor(float(total_pos), device=loss_total.device),
            "num_neg": torch.tensor(float(total_neg), device=loss_total.device),
        }

    @staticmethod
    def hard_negative_mining(loss: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, ratio: int = 3) -> torch.Tensor:

        num_pos = int(pos_mask.sum().item())
        num_neg_keep = min(int(neg_mask.sum().item()), ratio * max(num_pos, 1))
        if num_neg_keep == 0:
            return neg_mask.new_zeros(neg_mask.shape)
        # Get top-k negatives by loss
        neg_losses = loss.clone()
        neg_losses[~neg_mask] = -1.0  
        values, indices = torch.topk(neg_losses, k=num_neg_keep)
        selected = torch.zeros_like(neg_mask)
        selected[indices] = True
        selected = selected & neg_mask
        return selected
