import os
import tempfile
import pandas as pd
from surya.ocr import run_ocr
from surya.input.load import load_from_file
from surya.model.recognition.tokenizer import _tokenize
from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor



# Function to perform OCR on an image
def perform_ocr_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Define the language(s) for OCR
    langs = ["en"]  # English language
    # Load detection model and processor
    det_processor, det_model = load_detection_processor(), load_detection_model()
    # Load recognition model and processor
    rec_model, rec_processor = load_recognition_model(), load_recognition_processor()
    # Run OCR on the image
    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
    # Extract text from predictions
    data = []
    for txt in predictions:
        for txt_line in txt.text_lines:
            data.append(txt_line.text)
    return data

# Function to perform OCR on a PDF
def perform_ocr_pdf(pdf_path):
    # Load images from PDF
    images, _ = load_from_file(pdf_path, 1, 0)
    # Define the language(s) for OCR
    langs = ["en"]  # English language
    # Replace language names with language codes
    replace_lang_with_code(langs)
    # Determine languages for each image
    image_langs = [langs] * len(images)
    # Load detection processor
    det_processor = load_detection_processor()
    # Load detection model
    det_model = load_detection_model()
    # Tokenize language information
    _, lang_tokens = _tokenize("", get_unique_langs(image_langs))
    # Load recognition model with specified languages
    rec_model = load_recognition_model(langs=lang_tokens)
    # Load recognition processor
    rec_processor = load_recognition_processor()
    # Run OCR on each image
    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor)
    # Extract text from predictions
    data = []
    for txt in predictions_by_image:
        for txt_line in txt.text_lines:
            data.append(txt_line.text)
    return data

# Main part of the code to handle user input and call OCR functions
file_path = input("Enter the file path: ")

# Check if the provided file path exists
if os.path.isfile(file_path):
    # Determine the type of file and perform OCR accordingly
    if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
        # Perform OCR on image
        data = perform_ocr_image(file_path)
        print("""
        
        
        
        """)
        print("------------------OCR Result---------------------------")
        # Print OCR result
        for text in data:
            print(text)
    elif file_path.lower().endswith('pdf'):
        # Perform OCR on PDF
        data = perform_ocr_pdf(file_path)
        print("""
        
        
        
        """)
        print("------------------OCR Result---------------------------")
        # Print OCR result
        for text in data:
            print(text)
    else:
        print("Unsupported file format. Please upload an image (jpg, jpeg, png) or a PDF file.")
else:
    print("Invalid file path. Please provide a valid file path.")

