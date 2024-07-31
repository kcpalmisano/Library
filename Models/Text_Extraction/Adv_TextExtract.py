import cv2
import logging 
from PIL import Image
import numpy as np
import pytesseract

image_path = 'C:/Users/PATH/BC/BO_BC.jpg'

# Ensure tesseract is installed and accessible
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PATH\Tesseract-OCR\tesseract.exe'  # Update path to tesseract executable if needed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Load the image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use adaptive thresholding
binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Apply dilation and erosion to enhance text
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Sharpen the image
sharpen_kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1],
                           [-1, -1, -1]])
sharpened = cv2.filter2D(eroded, -1, sharpen_kernel)

# Convert the final processed image to PIL Image and display
final_image = Image.fromarray(sharpened)
final_image

# Use pytesseract to extract text
text = pytesseract.image_to_string(final_image)



#### Score text to the actual text
import Levenshtein

# Ground truth text (correct text)
ground_truth = "TEXT FROM THE IMAGE"

# Extracted text from OCR
ocr_text = text  # 'text' is the OCR output from pytesseract

# Calculate Character Error Rate (CER)
cer = Levenshtein.distance(ground_truth, ocr_text) / len(ground_truth)
print(f"Character Error Rate (CER): {cer:.2%}")

# Calculate Word Error Rate (WER)
def wer(ground_truth, ocr_text):
    ground_truth_words = ground_truth.split()
    ocr_text_words = ocr_text.split()
    word_dist = Levenshtein.distance(" ".join(ground_truth_words), " ".join(ocr_text_words))
    return word_dist / len(ground_truth_words)

wer_score = wer(ground_truth, ocr_text)
print(f"Word Error Rate (WER): {wer_score:.2%}")
