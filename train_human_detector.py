# train_human_detector_fast.py
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ---------------------------
# Paths
# ---------------------------
HUMAN_IMAGES = "datasets/humans/PNGImages"
HUMAN_MASKS  = "datasets/humans/PedMasks"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset
# ---------------------------
class PennFudanDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None, max_images=50):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = list(sorted(os.listdir(img_dir)))[:max_images]  # limit for speed
        self.masks = list(sorted(os.listdir(mask_dir)))[:max_images]
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)[1:]  # remove background
        masks = mask == obj_ids[:, None, None]

        boxes = []
        for m in masks:
            pos = np.where(m)
            boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # humans=1
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

# ---------------------------
# Transforms
# ---------------------------
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = PennFudanDataset(HUMAN_IMAGES, HUMAN_MASKS, transforms=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ---------------------------
# Model (MobileNet backbone for speed)
# ---------------------------
model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # background + human
model.to(device)

# ---------------------------
# Optimizer
# ---------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# ---------------------------
# Training
# ---------------------------
num_epochs = 2  # fast demo
for epoch in range(num_epochs):
    model.train()
    for imgs, targets in data_loader:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")

# ---------------------------
# Save Model
# ---------------------------
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "human_detector_fast.pth"))
print("[INFO] Fast human detector trained and saved!")
