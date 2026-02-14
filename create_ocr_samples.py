from PIL import Image, ImageDraw, ImageFont
import os

OCR_DIR = "datasets/OCR_test_images"
os.makedirs(OCR_DIR, exist_ok=True)

# Sample texts
texts = [
    "BOX 123\nA1B2C3",
    "MILITARY CONTAINER\nXYZ-789",
    "INDUSTRIAL PALLET\nCODE 4567"
]

for i, text in enumerate(texts, 1):
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))  # white background
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    draw.text((20, 50), text, fill=(0,0,0), font=font)
    img_path = os.path.join(OCR_DIR, f"ocr_sample_{i}.png")
    img.save(img_path)
    print(f"Created {img_path}")
