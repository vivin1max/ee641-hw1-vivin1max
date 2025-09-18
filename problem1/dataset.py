import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ShapeDetectionDataset(Dataset):

    def __init__(self, image_dir: str, annotation_file: str, transform: Optional[Any] = None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Map image_id -> file_name
        self.images: List[Dict[str, Any]] = data["images"]
        self.id_to_filename = {img["id"]: img["file_name"] for img in self.images}

        # Group annotations by image_id
        anns = data.get("annotations", [])
        self.img_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in anns:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Provide a deterministic index mapping
        self.index_ids = [img["id"] for img in self.images]

    def __len__(self) -> int:
        return len(self.index_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.index_ids[idx]
        file_name = self.id_to_filename[image_id]
        img_path = os.path.join(self.image_dir, file_name)

        # Load RGB image
        img = Image.open(img_path).convert("RGB")

        # Collect targets
        anns = self.img_to_anns.get(image_id, [])
        boxes = []
        labels = []
        for a in anns:
            bbox = a["bbox"]  
            boxes.append(bbox)
            labels.append(a["category_id"])  

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)
        else:
            
            np_img = np.array(img, dtype=np.uint8)
            img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0

        targets = {
            "boxes": boxes_t,
            "labels": labels_t,
        }
        return img, targets


def detection_collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    images = torch.stack(images, dim=0)
    return images, targets
