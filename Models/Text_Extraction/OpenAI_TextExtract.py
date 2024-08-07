import os
import cv2
import logging 
import pytesseract
from PIL import Image
from openai import OpenAI  
import openai
import time

client = OpenAI()

# Ensure tesseract is installed and accessible
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PATH\Tesseract-OCR\tesseract.exe'  # Update path to tesseract executable if needed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define the paths
input_folder = 'C:/Users/PATH/Data/BC/'  # replace with the path to your input folder
output_folder = 'C:/Users/PATH/Data/BCout/'  # replace with the path to your output folder
image_filename = 'BO_BC.jpg'  # replace with your image file name
output_filename = 'ext_text_test.txt'

# Construct the full paths
image_path = os.path.join(input_folder, image_filename)
output_path = os.path.join(output_folder, output_filename)

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Failed to load image: {image_path}")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding method
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert the processed image to PIL Image
    final_image = Image.fromarray(binary)

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(final_image)

    # Here, instead of printing, you would send `extracted_text` to ChatGPT.
    # For demonstration, we'll assume you have a function `send_to_chatgpt` that handles this.
    def send_to_chatgpt(extracted_text):
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Your system message or instructions"},
                        {"role": "user", "content": extracted_text},
                    ]
                )
                return response.choices[0].message['content']
            except openai.RateLimitError :
                retry_count += 1
                wait_time = 2 ** retry_count  # Exponential backoff
                time.sleep(wait_time)
                if retry_count == max_retries:
                    raise Exception("Rate limit exceeded. Max retries reached.")

    # Send extracted text to ChatGPT and get the response
    processed_text = send_to_chatgpt(extracted_text)

    # Save the processed text to a .txt file
    with open(output_path, 'w') as text_file:
        text_file.write(processed_text)

    print(f"Processed text saved to: {output_path}")
