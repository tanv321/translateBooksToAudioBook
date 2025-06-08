#!/usr/bin/env python3
"""
AI OCR Enhancer
--------------
This script uses OpenAI's API to enhance and correct OCR-extracted text from PDFs.
It processes the text to fix common OCR errors, restore missing words, and
improve formatting issues.
"""

import os
import argparse
import json
import time
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# Attempt to import from the pdf_to_text module
try:
    from pdf_to_text import process_pdf
except ImportError:
    print("Warning: pdf_to_text module not found. Please ensure it's in the same directory.")
    sys.exit(1)


class AiOcrEnhancer:
    def __init__(self, api_key=None, model="gpt-4", chunk_size=4000, overlap=200):
        """
        Initialize the AI OCR enhancer.
        
        Args:
            api_key: OpenAI API key (will also check OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4)
            chunk_size: Maximum size of text chunks to process at once
            overlap: Overlap between chunks to maintain context
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it with --api-key or OPENAI_API_KEY env variable.")
        
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def enhance_text(self, ocr_text, context=None):
        """
        Enhance OCR text using OpenAI to fix common OCR errors.
        
        Args:
            ocr_text: The raw OCR text to enhance
            context: Optional context about the document (title, author, subject, etc.)
            
        Returns:
            Enhanced text with corrections
        """
        # Split text into manageable chunks for API processing
        chunks = self._split_into_chunks(ocr_text)
        enhanced_chunks = []
        
        print(f"Processing {len(chunks)} text chunks...")
        
        # Process each chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Enhancing text")):
            # Add context for first chunk
            if i == 0 and context:
                system_prompt = f"""
                You are an expert OCR corrector. You fix errors in text extracted from PDFs.
                
                Document context:
                Title: {context.get('title', 'Unknown')}
                Author: {context.get('author', 'Unknown')}
                Subject: {context.get('subject', 'Unknown')}
                
                Fix these common OCR errors:
                1. Misrecognized characters and words
                2. Missing or extra spaces
                3. Improperly joined or split words
                4. Paragraph formatting issues
                5. Restore missing diacritical marks if appropriate
                
                Important guidelines:
                - Do NOT add or invent content that isn't implied by the text
                - DO correct obvious OCR errors based on context
                - Preserve the original paragraph structure and layout
                - If some text is completely unintelligible, leave it as is but add [?] after it
                - Fix obvious name misspellings
                """
            else:
                system_prompt = """
                You are an expert OCR corrector. You fix errors in text extracted from PDFs.
                
                Fix these common OCR errors:
                1. Misrecognized characters and words
                2. Missing or extra spaces
                3. Improperly joined or split words
                4. Paragraph formatting issues
                5. Restore missing diacritical marks if appropriate
                
                Important guidelines:
                - Do NOT add or invent content that isn't implied by the text
                - DO correct obvious OCR errors based on context
                - Preserve the original paragraph structure and layout
                - If some text is completely unintelligible, leave it as is but add [?] after it
                - Fix obvious name misspellings
                """
            
            user_prompt = f"Below is OCR-extracted text that may contain errors. Please correct it:\n\n{chunk}"
            
            # Add context for overlapping chunks after the first one
            if i > 0:
                user_prompt = f"This is a continuation of the previous text. Please correct any OCR errors:\n\n{chunk}"
            
            # Call OpenAI API
            response = self._call_openai_api(system_prompt, user_prompt)
            if response:
                enhanced_chunks.append(response)
            else:
                # If API call fails, keep original text
                enhanced_chunks.append(chunk)
                
            # Avoid rate limits
            time.sleep(0.5)
            
        # Combine enhanced chunks
        enhanced_text = self._combine_chunks(enhanced_chunks)
        return enhanced_text
    
    def _split_into_chunks(self, text):
        """Split text into overlapping chunks of appropriate size for the API."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position of current chunk
            end = start + self.chunk_size
            
            # Adjust to end at a paragraph boundary if possible
            if end < len(text):
                # Try to find paragraph break
                paragraph_end = text.rfind('\n\n', start, end)
                if paragraph_end > start + self.chunk_size // 2:
                    end = paragraph_end + 2  # Include the newlines
                else:
                    # If no paragraph break, try sentence break
                    sentence_end = text.rfind('. ', start, end)
                    if sentence_end > start + self.chunk_size // 2:
                        end = sentence_end + 2  # Include the period and space
            
            # Get chunk
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = max(start + 1, end - self.overlap)
            
            # Ensure we don't get stuck in a loop
            if start >= len(text):
                break
                
        return chunks
    
    def _combine_chunks(self, chunks):
        """
        Combine processed chunks back into full text, handling overlaps.
        This is a simple implementation - a more sophisticated approach would
        use diffing algorithms to identify and resolve conflicts.
        """
        if not chunks:
            return ""
        
        # For simplicity, just concatenate the chunks
        # In a more advanced implementation, we would need to handle
        # the overlapping regions more intelligently
        return "\n\n".join(chunks)
    
    def _call_openai_api(self, system_prompt, user_prompt):
        """Call OpenAI API with the given prompts."""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3,  # Lower temperature for more consistent corrections
                    "max_tokens": 4000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return None

    def process_pdf_file(self, pdf_path, output_path=None, metadata=None):
        """
        Process a PDF file, extract text, and enhance with AI.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path for the output enhanced text file
            metadata: Dictionary of metadata about the document
            
        Returns:
            Path to the enhanced text file
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        raw_text = process_pdf(pdf_path)
        
        if not raw_text:
            print("Failed to extract text from PDF.")
            return None
            
        # Set default output path if not provided
        if not output_path:
            output_path = os.path.splitext(pdf_path)[0] + "_enhanced.txt"
            
        # Enhance text with AI
        print("Enhancing text with AI...")
        enhanced_text = self.enhance_text(raw_text, context=metadata)
        
        # Save enhanced text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_text)
            
        print(f"Enhanced text saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Enhance OCR text with AI corrections')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output path for enhanced text')
    parser.add_argument('--api-key', help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4', help='OpenAI model to use (default: gpt-4)')
    parser.add_argument('--title', help='Document title for context')
    parser.add_argument('--author', help='Document author for context')
    parser.add_argument('--subject', help='Document subject for context')
    parser.add_argument('--chunk-size', type=int, default=4000, help='Size of text chunks to process')
    parser.add_argument('--overlap', type=int, default=200, help='Overlap between chunks')
    
    args = parser.parse_args()
    
    # Create metadata dictionary if any metadata is provided
    metadata = {}
    if args.title:
        metadata['title'] = args.title
    if args.author:
        metadata['author'] = args.author
    if args.subject:
        metadata['subject'] = args.subject
    
    # Initialize enhancer
    enhancer = AiOcrEnhancer(
        api_key=args.api_key,
        model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Process PDF
    enhancer.process_pdf_file(args.pdf_path, args.output, metadata)


if __name__ == "__main__":
    main()