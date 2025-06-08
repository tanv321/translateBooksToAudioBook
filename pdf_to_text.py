#!/usr/bin/env python3
"""
PDF to Text Converter with High Accuracy
----------------------------------------
This script converts PDF files to text using OCRmyPDF (for OCR if needed)
and PyPDF2 for text extraction from already-digital PDFs.
"""

import os
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import PyPDF2
import pytesseract
from pdf2image import convert_from_path


def check_requirements():
    """Check if required tools are installed."""
    try:
        # Check if OCRmyPDF is installed
        subprocess.run(["ocrmypdf", "--version"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("OCRmyPDF is not installed. Please install it with:")
        print("pip install ocrmypdf")
        return False
    
    try:
        # Check if Tesseract is installed
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed. Please install it.")
        print("On macOS: brew install tesseract")
        print("On Ubuntu: apt-get install tesseract-ocr")
        return False
    
    return True


def is_pdf_searchable(pdf_path):
    """Check if a PDF already contains searchable text."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        # Check first 3 pages (or all if fewer) for text
        num_pages_to_check = min(3, len(pdf_reader.pages))
        
        for i in range(num_pages_to_check):
            text = pdf_reader.pages[i].extract_text().strip()
            # If we find a reasonable amount of text, assume it's searchable
            if len(text) > 100:
                return True
    return False


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF that already has embedded text."""
    text_content = []
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        print(f"Extracting text from {num_pages} pages...")
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_content.append(text)
            if (page_num + 1) % 10 == 0:
                print(f"Processed {page_num + 1}/{num_pages} pages...")
    
    return "\n\n".join(text_content)


def ocr_pdf(input_pdf, output_pdf=None, language='eng'):
    """Run OCR on the PDF using OCRmyPDF."""
    if output_pdf is None:
        # Create a temporary file if no output is specified
        fd, output_pdf = tempfile.mkstemp(suffix='.pdf')
        os.close(fd)
        temp_created = True
    else:
        temp_created = False
    
    try:
        # Run OCRmyPDF with high quality settings
        cmd = [
            "ocrmypdf",
            "--force-ocr",               # Force OCR even if text exists
            "--deskew",                  # Fix skewed pages
            "--clean",                   # Clean images before OCR
            "--optimize", "3",           # Optimize PDF
            "--output-type", "pdf",      # Output PDF file
            "--skip-text",               # Skip pages with text
            "--language", language,      # OCR language
            "--jobs", "4",               # Number of parallel jobs
            input_pdf,                   # Input file
            output_pdf                   # Output file
        ]
        
        print(f"Running OCR with language '{language}'...")
        subprocess.run(cmd, check=True)
        
        # Extract text from the OCRed PDF
        text = extract_text_from_pdf(output_pdf)
        
        # Clean up temp file if we created one
        if temp_created:
            os.unlink(output_pdf)
            
        return text
    
    except subprocess.CalledProcessError as e:
        print(f"Error during OCR: {e}")
        if temp_created and os.path.exists(output_pdf):
            os.unlink(output_pdf)
        return None


def pdf_to_text_tesseract(pdf_path, language='eng', dpi=300):
    """Convert PDF to text using pdf2image and Tesseract directly."""
    # Create a temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Convert PDF to images
        print(f"Converting PDF to images at {dpi} DPI...")
        images = convert_from_path(pdf_path, dpi=dpi)
        
        # Process each image with Tesseract
        text_content = []
        for i, image in enumerate(images):
            print(f"OCR processing page {i+1}/{len(images)}...")
            
            # Save the image to the temp directory
            image_path = os.path.join(temp_dir, f'page_{i+1}.png')
            image.save(image_path, 'PNG')
            
            # Run Tesseract on the image
            text = pytesseract.image_to_string(image_path, lang=language)
            text_content.append(text)
            
            # Remove the image file
            os.unlink(image_path)
        
        return "\n\n".join(text_content)
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def process_pdf(pdf_path, output_text_path=None, language='eng', force_ocr=False, method='auto'):
    """
    Process a PDF file to extract text with high accuracy.
    
    Args:
        pdf_path: Path to the PDF file
        output_text_path: Path to save the extracted text
        language: Language for OCR (default: English)
        force_ocr: Force OCR even if the PDF is already searchable
        method: OCR method to use ('auto', 'ocrmypdf', or 'tesseract')
    
    Returns:
        The extracted text
    """
    pdf_path = os.path.abspath(pdf_path)
    
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return None
    
    print(f"Processing {pdf_path}...")
    
    # Check if the PDF is already searchable
    if not force_ocr and method == 'auto' and is_pdf_searchable(pdf_path):
        print("PDF already contains searchable text. Extracting...")
        text = extract_text_from_pdf(pdf_path)
    else:
        # Perform OCR using the specified method
        if method == 'tesseract' or (method == 'auto' and shutil.which('ocrmypdf') is None):
            print("Using Tesseract OCR directly...")
            text = pdf_to_text_tesseract(pdf_path, language=language)
        else:
            print("Using OCRmyPDF for processing...")
            text = ocr_pdf(pdf_path, language=language)
    
    # Save the text to a file if requested
    if output_text_path and text:
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text saved to {output_text_path}")
    
    return text


def main():
    parser = argparse.ArgumentParser(description='Convert PDF to text with high accuracy.')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output text file path (default: same name as PDF with .txt extension)')
    parser.add_argument('-l', '--language', default='eng', help='OCR language (default: eng)')
    parser.add_argument('-f', '--force-ocr', action='store_true', help='Force OCR even if the PDF is already searchable')
    parser.add_argument('-m', '--method', choices=['auto', 'ocrmypdf', 'tesseract'], default='auto',
                        help='OCR method to use (default: auto)')
    
    args = parser.parse_args()
    
    # Check if the required tools are installed
    if not check_requirements():
        return
    
    pdf_path = args.pdf_path
    
    # Set default output path if not specified
    if not args.output:
        output_path = os.path.splitext(pdf_path)[0] + '.txt'
    else:
        output_path = args.output
    
    # Process the PDF
    process_pdf(pdf_path, output_path, args.language, args.force_ocr, args.method)


if __name__ == "__main__":
    main()