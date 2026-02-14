"""
AI Technical Assignment: Human & Animal Detection + Offline OCR + Video
Author: Mandapalli Usha
Date: 14-02-2026

This script performs:
1. Human detection using MTCNN
2. Animal classification using ResNet18
3. Offline OCR on industrial/stenciled text images using Tesseract
4. Video processing for human/animal detection
"""

import os
import glob
import random
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN
import cv2
import pytesseract
import json
import numpy as np

# ---------------------------
# Step 0: Setup paths and device
# ---------------------------
ANIMAL_MODEL_PATH = "models/animal_classifier.pth"
ANIMAL_TEST_DIR   = "datasets/Animals/raw-img/test"
HUMAN_TEST_DIR    = "datasets/humans/PNGImages"
OCR_DIR           = "datasets/OCR_test_images"
RESULTS_DIR       = "results"
TEST_VIDEOS_DIR   = "test_videos"
OUTPUT_VIDEOS_DIR = "outputs"

os.makedirs(OCR_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Step 1: Create sample OCR images
# ---------------------------
texts = [
    "BOX 123\nA1B2C3",
    "MILITARY CONTAINER\nXYZ-789",
    "INDUSTRIAL PALLET\nCODE 4567"
]

for i, text in enumerate(texts, 1):
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    draw.text((20, 50), text, fill=(0,0,0), font=font)
    img_path = os.path.join(OCR_DIR, f"ocr_sample_{i}.png")
    img.save(img_path)
    print(f"[INFO] Created OCR sample image: {img_path}")

# ---------------------------
# Step 2: Load animal classifier
# ---------------------------
animal_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
animal_model.fc = torch.nn.Linear(animal_model.fc.in_features, 2)  # two classes: cane, cavallo
animal_model.load_state_dict(torch.load(ANIMAL_MODEL_PATH, map_location=device))
animal_model.to(device)
animal_model.eval()
ANIMAL_CLASSES = ["cane", "cavallo"]

animal_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ---------------------------
# Step 3: Load human face detector
# ---------------------------
mtcnn = MTCNN(keep_all=True, device=device)

# ---------------------------
# Step 4: Helper functions
# ---------------------------
def predict_animal(image):
    img_tensor = animal_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = animal_model(img_tensor)
        _, pred = torch.max(outputs, 1)
    return ANIMAL_CLASSES[pred.item()]

def detect_faces(image):
    boxes, _ = mtcnn.detect(image)
    return boxes

def annotate_image(image_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Detect humans
    boxes = detect_faces(img)
    human_detected = boxes is not None and len(boxes) > 0
    if human_detected:
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
            draw.text((x1, y1-30), "Human", fill="green", font=font)

    # Detect animals only if no humans detected
    if not human_detected:
        animal_class = predict_animal(img)
        w, h = img.size
        draw.rectangle([0, 0, w, h], outline="red", width=4)
        draw.text((10, 10), f"Animal: {animal_class}", fill="red", font=font)

    save_path = os.path.join(RESULTS_DIR, os.path.basename(image_path))
    img.save(save_path)
    return save_path

# ---------------------------
# Step 5: Annotate random 10 images each
# ---------------------------
# Annotate animal and human images
image_paths = glob.glob(os.path.join(ANIMAL_TEST_DIR, "*", "*.png")) + \
              glob.glob(os.path.join(HUMAN_TEST_DIR, "**", "*.png"), recursive=True)
random_images = random.sample(image_paths, min(20, len(image_paths)))

print("[INFO] Annotating random images...")
for img_path in random_images:
    saved_path = annotate_image(img_path)
    print(f"{img_path} -> saved to {saved_path}")

# ---------------------------
# Step 6: Offline OCR
# ---------------------------
OCR_OUTPUT_FILE = os.path.join(RESULTS_DIR,"ocr_results.json")

def preprocess_ocr(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thres = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thres

def ocr_pipeline(input_folder=OCR_DIR, output_file=OCR_OUTPUT_FILE):
    results = {}
    for fname in os.listdir(input_folder):
        if fname.lower().endswith((".png",".jpg",".jpeg")):
            img_path = os.path.join(input_folder,fname)
            preprocessed = preprocess_ocr(img_path)
            text = pytesseract.image_to_string(preprocessed, config="--psm 6")
            results[fname] = text.strip()
    with open(output_file,'w') as f:
        json.dump(results,f,indent=4)
    print(f"[INFO] OCR complete! Results saved to {output_file}")

ocr_pipeline()

# ---------------------------
# Step 7: Video processing for human & animal detection
# ---------------------------
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 5

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Detect humans
        boxes = detect_faces(pil_frame)
        human_detected = boxes is not None and len(boxes) > 0
        if human_detected:
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
                draw.text((x1, y1-30), "Human", fill="green", font=font)

        # Detect animals only if no humans detected
        if not human_detected:
            animal_class = predict_animal(pil_frame)
            w, h = pil_frame.size
            draw.rectangle([0, 0, w, h], outline="red", width=4)
            draw.text((10, 10), f"Animal: {animal_class}", fill="red", font=font)

        annotated_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"[INFO] Processed video saved: {output_path}")

# Process all videos in test_videos/
video_files = [f for f in os.listdir(TEST_VIDEOS_DIR) if f.lower().endswith((".mp4",".avi"))]
for vid_file in video_files:
    input_path = os.path.join(TEST_VIDEOS_DIR, vid_file)
    output_path = os.path.join(OUTPUT_VIDEOS_DIR, f"annotated_{vid_file}")
    process_video(input_path, output_path)

print("[INFO] All done! Check results folder for images & OCR, outputs folder for videos.")
