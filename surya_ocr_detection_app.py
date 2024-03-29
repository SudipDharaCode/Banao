import streamlit as st
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor

def process_image_and_get_text(image_path):
    image = Image.open(image_path)
    langs = ["en"]  # Replace with your languages
    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()

    predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

    data = []
    for txt in predictions:
        for txt_line in txt.text_lines:
            data.append(txt_line.text)
    return data

def main():
    st.title("Image Upload and OCR App")
    
    # Display upload option
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Process the uploaded image and get text
        texts = process_image_and_get_text(uploaded_file)

        # Display the processed text
        st.header("Processed Text:")
        for text in texts:
            st.write(text)

if __name__ == "__main__":
    main()
