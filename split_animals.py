# split_animals.py
import os
import shutil
import random

# Path to your animals folder (with cane/ and cavallo/)
src = "datasets/Animals/raw-img"
classes = ["cane", "cavallo"]

for cls in classes:
    files = os.listdir(os.path.join(src, cls))
    random.shuffle(files)
    
    # Split 80% train, 20% test
    train_count = int(0.8 * len(files))
    
    # Create train/test folders
    train_dir = os.path.join(src, "train", cls)
    test_dir  = os.path.join(src, "test", cls)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy files
    for i, f in enumerate(files):
        if i < train_count:
            shutil.copy(os.path.join(src, cls, f), os.path.join(train_dir, f))
        else:
            shutil.copy(os.path.join(src, cls, f), os.path.join(test_dir, f))

print("✅ Train/test split completed!")
