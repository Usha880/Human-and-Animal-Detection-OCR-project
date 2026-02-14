# AI Technical Assignment: Human & Animal Detection + Offline OCR

**Author:** Mandapalli Usha  
**Date:** 14-02-2026  

This project demonstrates a fully **offline AI system** combining computer vision and OCR for real-world industrial applications. It detects humans and animals in images/videos and extracts text from industrial/stenciled boxes.

---

## **Project Overview**

The project is divided into two main parts:

### **Part A: Human & Animal Detection**
- Detect humans using **MTCNN (Multi-task Cascaded Convolutional Network)**.
- Detect and classify animals using **ResNet18** (two classes: `cane`, `cavallo`).
- Annotate images and videos with bounding boxes and class labels:
  - **Humans:** Green boxes
  - **Animals:** Red boxes
- Video processing detects humans and animals frame by frame and outputs annotated videos.

### **Part B: Offline OCR for Industrial Text**
- Offline OCR on stenciled or painted text on boxes (faded paint, low contrast, surface damage).
- Preprocessing includes:
  - Grayscale conversion
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Thresholding
- Extracted text saved as structured **JSON**.
- Fully offline using **Tesseract OCR** (no cloud APIs).

---

## **Directory Structure**



human_animals_project/

├── datasets/

│ ├── Animals/raw-img/test/

│ ├── humans/PNGImages/

│ └── OCR_test_images/

├── models/

│ └── animal_classifier.pth

├── results/

│ ├── annotated images (.png)

│ └── ocr_results.json

├── test_videos/

│ └── sample videos (.mp4)

├── outputs/

│ └── annotated videos (.mp4)

├── main.py

├── streamlit_app.py

├── requirements.txt

└── README.md


---

## **Dependencies**

Python 3.10+ and the following packages:

```bash
torch
torchvision
facenet-pytorch
pillow
opencv-python
pytesseract
streamlit
numpy


Note: Tesseract OCR must be installed locally on your machine and accessible in PATH.

How to Run
1. Image Annotation
python main.py


Annotates 10 random human and animal images from datasets.

Saves annotated images in results/.

2. Video Annotation

Place videos in test_videos/.

main.py processes all videos and saves annotated versions in outputs/.

3. Offline OCR

OCR images are in datasets/OCR_test_images/.

Extracted text saved to results/ocr_results.json.

4. Streamlit Visualization
streamlit run streamlit_app.py


Interactive interface to view:

Annotated images

OCR text results

Sample Outputs

Annotated Images:




Video Outputs:




Replace the above with your actual screenshots in the screenshots/ folder.

Challenges & Improvements
Challenges

Faded or low-contrast text in OCR images.

Partially visible humans may not be detected by MTCNN.

Small dataset for animals limits classification accuracy.

Possible Improvements

Expand animal dataset with more classes and images.

Use full-body human detection (Faster-RCNN) for better coverage.

Apply advanced OCR (EAST or deep learning-based) for industrial text.

Data augmentation to improve OCR accuracy.

Conclusion

This project demonstrates a full offline AI pipeline for:

Human detection via MTCNN

Animal classification via ResNet18

Industrial OCR with Tesseract

All outputs are structured, visualized, and ready for offline deployment.
