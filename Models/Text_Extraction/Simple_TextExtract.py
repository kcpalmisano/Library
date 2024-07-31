
import os
import logging
import cv2
import pytesseract
from PIL import Image
import pdfplumber
from googletrans import Translator
import argparse
import sys
import numpy as np
import Levenshtein
from textblob import TextBlob
from collections import Counter

# Ensure tesseract is installed and accessible
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PATH\Tesseract-OCR\tesseract.exe'  # Update path to tesseract executable if needed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_image(image_path):
    """
    Preprocesses an image to enhance the quality of text extraction.
    - Converts the image to grayscale.
    - Applies simple thresholding to handle varying lighting conditions.

    :param image_path: Path to the image file.
    :return: Preprocessed image object or None if an error occurs.
    """
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

def extract_text_from_image(image_path, custom_config):
    """
    Extracts text from an image file using Tesseract-OCR after preprocessing the image.

    :param image_path: Path to the image file.
    :param custom_config: Tesseract configuration string.
    :return: Extracted text as a string, or an empty string if an error occurs.
    """
    try:
        image = preprocess_image(image_path)
        if image:
            text = pytesseract.image_to_string(image, config=custom_config)
            return text
        else:
            return ""
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfplumber.

    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a string, or an empty string if an error occurs.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def translate_text(text, target_language='en'):
    """
    Translates the given text to the specified target language using googletrans.

    :param text: Text to be translated.
    :param target_language: Target language code (default is 'en' for English).
    :return: Translated text as a string, or the original text if an error occurs.
    """
    try:
        translator = Translator()
        translated = translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return text

def organize_text_lines(text):
    """
    Organizes text into clean lines by removing extra whitespace and empty lines.

    :param text: Text to be organized.
    :return: Organized text as a single string with clean lines.
    """
    lines = text.split('\n')
    organized_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(organized_lines)

def write_to_file(text, output_path):
    """
    Writes the given text to a specified file.

    :param text: Text to be written to the file.
    :param output_path: Path to the output file.
    """
    try:
        with open(output_path, 'w') as file:
            file.write(text)
        logging.info(f"Successfully wrote to {output_path}")
    except Exception as e:
        logging.error(f"Error writing to file: {e}")

def calculate_similarity(text1, text2):
    """
    Calculates the similarity between two texts using Levenshtein distance.

    :param text1: First text string.
    :param text2: Second text string.
    :return: Similarity score (lower is better).
    """
    return Levenshtein.distance(text1, text2)

def combine_texts(texts):
    """
    Combines multiple text outputs using a voting mechanism to select the most frequent words.
    
    :param texts: List of text strings.
    :return: Combined text string.
    """
    lines = [text.split('\n') for text in texts]
    combined_lines = []

    for lines_group in zip(*lines):
        words = [line.split() for line in lines_group]
        combined_words = []

        for words_group in zip(*words):
            most_common_word, _ = Counter(words_group).most_common(1)[0]
            combined_words.append(most_common_word)

        combined_lines.append(' '.join(combined_words))

    return '\n'.join(combined_lines)

def get_best_text(texts):
    """
    Selects the best text from a list of texts based on similarity.

    :param texts: List of text strings.
    :return: Best text string.
    """
    if not texts:
        return ""

    base_text = texts[0]
    best_text = base_text
    best_score = float('inf')

    for i, text in enumerate(texts):
        score = calculate_similarity(base_text, text)
        print(f"Levenshtein distance between base text and text {i+1}: {score}")
        if score < best_score:
            best_text = text
            best_score = score

    return best_text

def correct_text(text):
    """
    Corrects the text using basic NLP techniques.

    :param text: Text to be corrected.
    :return: Corrected text.
    """
    corrected_text = str(TextBlob(text).correct())
    return corrected_text

def process_file(input_path, output_path, file_type='image'):
    """
    Processes a single file (image or PDF), extracts, translates, organizes text, and writes it to an output file.

    :param input_path: Path to the input file.
    :param output_path: Path to the output file.
    :param file_type: Type of the input file ('image' or 'pdf').
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file does not exist: {input_path}")
        return

    # Define configurations
    custom_config_typed = r'--oem 1 --psm 6'
    custom_config_handwritten = r'--oem 1 --psm 11'
    custom_config_mixed = r'--oem 3 --psm 3'

    if file_type == 'image':
        # Extract text using different configurations
        text_typed = extract_text_from_image(input_path, custom_config_typed)
        text_handwritten = extract_text_from_image(input_path, custom_config_handwritten)
        text_mixed = extract_text_from_image(input_path, custom_config_mixed)

        # Combine the texts from different configurations
        combined_text = combine_texts([text_typed, text_handwritten, text_mixed])

        # Get the best text from the different configurations
        best_text = get_best_text([text_typed, text_handwritten, text_mixed])

        # Correct the best text using NLP
        corrected_text = correct_text(best_text)
    elif file_type == 'pdf':
        combined_text = extract_text_from_pdf(input_path)
        corrected_text = correct_text(combined_text)
    else:
        logging.error("Unsupported file type. Use 'image' or 'pdf'.")
        return

    if not corrected_text:
        logging.warning("No text extracted.")
        return

    translated_text = translate_text(corrected_text)
    organized_text = organize_text_lines(translated_text)
    write_to_file(organized_text, output_path)

def process_directory(input_dir, output_dir, file_type='image'):
    """
    Processes all files in a directory, extracts, translates, organizes text from each file, and writes it to corresponding output files.

    :param input_dir: Path to the input directory containing files.
    :param output_dir: Path to the output directory where results will be saved.
    :param file_type: Type of the input files ('image' or 'pdf').
    """
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        process_file(input_path, output_path, file_type)

def main():
    """
    Main function to handle command-line arguments and initiate the processing of files or directories.
    """
    parser = argparse.ArgumentParser(description="Extract, translate, and organize text from images or PDFs.")
    parser.add_argument('input_path', help="Path to the input file or directory")
    parser.add_argument('output_path', help="Path to the output file or directory")
    parser.add_argument('--file_type', choices=['image', 'pdf'], default='image', help="Type of the input files")
    parser.add_argument('--is_directory', action='store_true', help="Indicates if the input path is a directory")

    # Use sys.argv for interactive environments like Jupyter or Spyder
    if len(sys.argv) == 1:
        # No command-line arguments provided, set default paths for testing
        args = parser.parse_args([
           'C:/Users/PATH/Data/BC/', #Directory
          ##'C:/Users/PATH/Data/BC/BO_BC.jpg',  #single image
           'C:/Users/PATH/Data/BCout/',
           '--file_type', 'image'
            , '--is_directory'  # Indicating it is a directory
        ])
    else:
        args = parser.parse_args()

    if args.is_directory:
        process_directory(args.input_path, args.output_path, args.file_type)
    else:
        process_file(args.input_path, args.output_path, args.file_type)


if __name__ == "__main__":
    main()
