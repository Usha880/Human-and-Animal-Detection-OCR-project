import cv2
import glob
import os
import random

# ---------------------------
# Paths (based on your setup)
# ---------------------------
HUMAN_IMAGES_DIR  = "datasets/humans/PNGImages"
ANIMAL_IMAGES_DIR = "datasets/Animals/raw-img"
TEST_VIDEOS_DIR   = "test_videos"

os.makedirs(TEST_VIDEOS_DIR, exist_ok=True)

# ---------------------------
# Collect human and animal images
# ---------------------------
human_imgs = glob.glob(os.path.join(HUMAN_IMAGES_DIR, "*.png")) + \
             glob.glob(os.path.join(HUMAN_IMAGES_DIR, "*.jpg")) + \
             glob.glob(os.path.join(HUMAN_IMAGES_DIR, "*.jpeg"))

animal_imgs = glob.glob(os.path.join(ANIMAL_IMAGES_DIR, "*/*.png")) + \
              glob.glob(os.path.join(ANIMAL_IMAGES_DIR, "*/*.jpg")) + \
              glob.glob(os.path.join(ANIMAL_IMAGES_DIR, "*/*.jpeg"))

if len(human_imgs) == 0 or len(animal_imgs) == 0:
    raise ValueError("No human or animal images found. Check your folder paths!")

# ---------------------------
# Create 2 mixed videos
# ---------------------------
for vid_num in range(1, 3):
    # Sample 10 humans and 10 animals
    human_sample = random.sample(human_imgs, min(10, len(human_imgs)))
    animal_sample = random.sample(animal_imgs, min(10, len(animal_imgs)))
    
    # Alternate frames: human -> animal
    frames_paths = []
    for h, a in zip(human_sample, animal_sample):
        frames_paths.append(("human", h))
        frames_paths.append(("animal", a))
    
    # Add remaining frames if unequal
    if len(human_sample) > len(animal_sample):
        for h in human_sample[len(animal_sample):]:
            frames_paths.append(("human", h))
    elif len(animal_sample) > len(human_sample):
        for a in animal_sample[len(human_sample):]:
            frames_paths.append(("animal", a))
    
    # Video writer
    width, height = 256, 256
    out_path = os.path.join(TEST_VIDEOS_DIR, f"mixed_video_{vid_num}.mp4")
    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          2,  # fps
                          (width, height))
    
    for label, img_path in frames_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        out.write(img)
    
    out.release()
    print(f"[INFO] Created {out_path} with {len(frames_paths)} frames (humans + animals)")
