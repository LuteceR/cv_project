"""
👍
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image

from segmentModel import HandRecogniionModel

DATASET_PATH = "eyth_dataset"

# model = HandRecogniionModel(
#     n_channels=16,
#     n_classes=1,
#     )

image = Image.open("eyth_dataset/masks/vid9/frame900.png")

print(np.array(image, dtype=np.uint8))