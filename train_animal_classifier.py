# train_animal_classifier.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------
# Paths
# ---------------------------
ANIMAL_DATASET = "datasets/Animals/raw-img"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ---------------------------
# Dataset
# ---------------------------
train_dataset = datasets.ImageFolder(os.path.join(ANIMAL_DATASET, "train"), transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(ANIMAL_DATASET, "test"), transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader   = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------
# Model
# ---------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cane, cavallo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 3

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ---------------------------
# Save Model
# ---------------------------
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "animal_classifier.pth"))
print("[INFO] Animal classifier trained and saved!")
