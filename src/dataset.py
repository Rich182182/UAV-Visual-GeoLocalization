import math
from PIL import Image
import torch
from torch.utils.data import Dataset


class UAVTripletDataset(Dataset):
    """Dataset yielding (anchor, positive, negative, rotation_sincos) tuples."""

    def __init__(self, pairs, transform_drone, transform_sat):
        self.pairs = pairs
        self.transform_drone = transform_drone
        self.transform_sat = transform_sat

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        anchor_img = Image.open(pair["anchor"]).convert("RGB")
        positive_img = Image.open(pair["positive"]).convert("RGB")
        negative_img = Image.open(pair["negative"]).convert("RGB")

        anchor = self.transform_drone(anchor_img)
        positive = self.transform_sat(positive_img)
        negative = self.transform_sat(negative_img)

        kappa_rad = pair["kappa"]
        rot_sincos = torch.tensor(
            [math.sin(kappa_rad), math.cos(kappa_rad)],
            dtype=torch.float32,
        )

        return anchor, positive, negative, rot_sincos