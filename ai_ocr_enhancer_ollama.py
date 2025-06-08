#!/usr/bin/env python3
"""
AI Text Enhancer for PDFs and EPUBs
-----------------------------------
This script uses AI (OpenAI or Ollama) to enhance and correct text extracted from PDFs and EPUBs.
It processes the text to fix common OCR errors, restore missing words, improve formatting,
and filters out non-content elements like page numbers and headers.
"""

import os
import argparse
import json
import time
import sys
import re
from pathlib import Path
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import html2text

load_dotenv()

# Attempt to import from the pdf_to_text module
try:
    from pdf_to_text import process_pdf
except ImportError:
    print("Warning: pdf_to_text module not found. PDF support will be limited.")
    process_pdf = None


class TextContentFilter:
    """Filter out non-content elements from extracted text."""
    
    @staticmethod
    def is_page_number(line):
        """Check if a line is likely just a page number."""
        line = line.strip()
        # Common page number patterns
        patterns = [
            r'^\d{1,4}$',  # Just numbers
            r'^Page\s+\d+$',  # Page 123
            r'^\d+\s*of\s*\d+$',  # 12 of 345
            r'^-\s*\d+\s*-$',  # - 123 -
            r'^\[\s*\d+\s*\]$',  # [123]
        ]
        return any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns)
    
    @staticmethod
    def is_header_footer(line, position_in_page=None):
        """Check if a line is likely a header or footer."""
        line = line.strip()
        # Very short lines at the beginning or end of pages are often headers/footers
        if len(line) < 5:
            return True
        # Running headers often have specific patterns
        if re.match(r'^Chapter\s+\d+$', line, re.IGNORECASE):
            return True
        if re.match(r'^\d+\s*[|/\\]\s*\w+', line):  # Like "12 | Chapter Name"
            return True
        return False
    
    @staticmethod
    def clean_text_content(text):
        """Remove common non-content elements from text."""
        lines = text.split('\n')
        cleaned_lines = []
        
        # Track patterns to identify headers/footers
        page_breaks = []
        for i, line in enumerate(lines):
            if re.match(r'^={3,}$|^-{3,}$|^\*{3,}$', line.strip()):
                page_breaks.append(i)
        
        for i, line in enumerate(lines):
            # Skip if it's a page number
            if TextContentFilter.is_page_number(line):
                continue
            
            # Skip if it's likely a header/footer
            if TextContentFilter.is_header_footer(line):
                continue
            
            # Skip lines that are just separators
            if re.match(r'^[\s\-=_*]{3,}$', line):
                continue
            
            cleaned_lines.append(line)
        
        # Rejoin and clean up excessive whitespace
        cleaned_text = '\n'.join(cleaned_lines)
        # Replace multiple newlines with double newlines for paragraphs
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text


class EpubProcessor:
    """Process EPUB files and extract text content."""
    
    def __init__(self):
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = False
        self.h2t.body_width = 0  # Don't wrap text
        
    def extract_text_from_epub(self, epub_path):
        """Extract text content from an EPUB file."""
        try:
            book = epub.read_epub(epub_path)
            text_content = []
            
            # Extract metadata
            metadata = {
                'title': book.get_metadata('DC', 'title'),
                'author': book.get_metadata('DC', 'creator'),
                'subject': book.get_metadata('DC', 'subject')
            }
            
            # Clean metadata
            for key in metadata:
                if metadata[key] and isinstance(metadata[key], list):
                    metadata[key] = metadata[key][0][0] if metadata[key][0] else None
            
            # Process each item in the EPUB
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Convert to text
                    text = self.h2t.handle(str(soup))
                    
                    # Clean up the text
                    text = TextContentFilter.clean_text_content(text)
                    
                    if text.strip():
                        text_content.append(text)
            
            full_text = '\n\n'.join(text_content)
            return full_text, metadata
            
        except Exception as e:
            print(f"Error processing EPUB: {str(e)}")
            return None, None


class AiTextEnhancer:
    def __init__(self, provider="ollama", api_key=None, model=None, chunk_size=4000, overlap=200):
        """
        Initialize the AI text enhancer.
        
        Args:
            provider: 'ollama' for local inference or 'openai' for OpenAI API
            api_key: API key (required for OpenAI)
            model: Model to use (default: mistral:latest for Ollama, gpt-4 for OpenAI)
            chunk_size: Maximum size of text chunks to process at once
            overlap: Overlap between chunks to maintain context
        """
        self.provider = provider
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.epub_processor = EpubProcessor()
        
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set it with --api-key or OPENAI_API_KEY env variable.")
            self.model = model or "gpt-4"
        else:  # ollama
            self.model = model or "mistral:latest"
            # Test Ollama connection
            self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = [m['name'] for m in response.json()['models']]
                if self.model not in models:
                    print(f"\nWarning: Model '{self.model}' not found.")
                    print(f"Available models: {models}")
                    print(f"To download the model, run: ollama pull {self.model}\n")
                else:
                    print(f"✓ Ollama is running with {self.model}")
            else:
                print("Error: Ollama is not responding properly.")
        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to Ollama.")
            print("Make sure Ollama is running with: ollama serve")
            print("If Ollama is not installed, visit: https://ollama.ai\n")
            sys.exit(1)
    
    def _call_ollama_api(self, system_prompt, user_prompt):
        """Call local Ollama API."""
        try:
            # Combine prompts for Mistral
            full_prompt = f"""<s>[INST] {system_prompt}

{user_prompt} [/INST]"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 4096,
                        "stop": ["</s>", "[INST]", "[/INST]"]
                    }
                },
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            print("Warning: Ollama request timed out. Consider using smaller chunks.")
            return None
        except Exception as e:
            print(f"Error calling Ollama API: {str(e)}")
            return None
    
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
                    "temperature": 0.3,
                    "max_tokens": 4000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return None
    
    def _call_api(self, system_prompt, user_prompt):
        """Unified API calling method."""
        if self.provider == "ollama":
            return self._call_ollama_api(system_prompt, user_prompt)
        else:
            return self._call_openai_api(system_prompt, user_prompt)
        
    def enhance_text(self, text, context=None, is_ocr=True):
        """
        Enhance text using AI to fix errors and improve readability.
        
        Args:
            text: The raw text to enhance
            context: Optional context about the document (title, author, subject, etc.)
            is_ocr: Whether the text is from OCR (has different types of errors)
            
        Returns:
            Enhanced text with corrections
        """
        # First, apply content filtering
        text = TextContentFilter.clean_text_content(text)
        
        # Split text into manageable chunks for API processing
        chunks = self._split_into_chunks(text)
        enhanced_chunks = []
        
        print(f"\nProcessing {len(chunks)} text chunks with {self.model}...")
        if self.provider == "ollama":
            print("(This may take a few minutes for local processing)")
        
        # Process each chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Enhancing text")):
            # Adjust system prompt based on whether it's OCR or EPUB
            if is_ocr:
                system_prompt = self._get_ocr_system_prompt(context, i == 0)
            else:
                system_prompt = self._get_epub_system_prompt(context, i == 0)
            
            user_prompt = f"Below is extracted text that may contain errors or formatting issues. Please correct it:\n\n{chunk}"
            
            # Add context for overlapping chunks after the first one
            if i > 0:
                user_prompt = f"This is a continuation of the previous text. Please correct any errors:\n\n{chunk}"
            
            # Call API
            response = self._call_api(system_prompt, user_prompt)
            if response:
                enhanced_chunks.append(response)
            else:
                # If API call fails, keep original text
                enhanced_chunks.append(chunk)
                print(f"\nWarning: Chunk {i+1} enhancement failed, keeping original")
                
            # Rate limiting only for OpenAI
            if self.provider == "openai":
                time.sleep(0.5)
            
        # Combine enhanced chunks
        enhanced_text = self._combine_chunks(enhanced_chunks)
        return enhanced_text
    
    def _get_ocr_system_prompt(self, context, is_first_chunk):
        """Get system prompt for OCR text enhancement."""
        base_prompt = """You are an expert text corrector specializing in fixing OCR errors from scanned books and documents.
        
Fix these common OCR errors:
1. Misrecognized characters and words
2. Missing or extra spaces
3. Improperly joined or split words
4. Paragraph formatting issues
5. Restore missing diacritical marks if appropriate
6. Fix scrambled footnotes that appear in the wrong place

Important guidelines:
- Do NOT add or invent content that isn't implied by the text
- DO correct obvious OCR errors based on context
- Preserve the original paragraph structure and layout
- If some text is completely unintelligible, leave it as is but add [?] after it
- Fix obvious name misspellings
- Maintain footnotes in their logical positions
- Remove any obvious page numbers or headers that don't belong to the main text"""
        
        if is_first_chunk and context:
            context_info = f"""Document context:
Title: {context.get('title', 'Unknown')}
Author: {context.get('author', 'Unknown')}
Subject: {context.get('subject', 'Unknown')}
"""
            return context_info + "\n" + base_prompt
        
        return base_prompt
    
    def _get_epub_system_prompt(self, context, is_first_chunk):
        """Get system prompt for EPUB text enhancement."""
        base_prompt = """You are an expert text editor working with digital book content.
        
Your tasks:
1. Fix any formatting issues from HTML to text conversion
2. Ensure proper paragraph breaks
3. Fix any encoding errors or special character issues
4. Maintain emphasis (italics/bold) where appropriate using markdown
5. Ensure footnotes are properly positioned
6. Remove any artifacts from the conversion process

Important guidelines:
- Preserve the author's original text and intent
- Maintain the book's structure and flow
- Keep chapter divisions clear
- Do not add content that isn't in the original
- Fix only technical/formatting issues, not style"""
        
        if is_first_chunk and context:
            context_info = f"""Book information:
Title: {context.get('title', 'Unknown')}
Author: {context.get('author', 'Unknown')}
Subject: {context.get('subject', 'Unknown')}
"""
            return context_info + "\n" + base_prompt
        
        return base_prompt
    
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
        """
        if not chunks:
            return ""
        
        # For simplicity, just concatenate the chunks
        # In a more advanced implementation, we would need to handle
        # the overlapping regions more intelligently
        return "\n\n".join(chunks)

    def process_file(self, file_path, output_path=None, metadata=None):
        """
        Process a PDF or EPUB file, extract text, and enhance with AI.
        
        Args:
            file_path: Path to the file (PDF or EPUB)
            output_path: Path for the output enhanced text file
            metadata: Dictionary of metadata about the document
            
        Returns:
            Path to the enhanced text file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
            
        print(f"\nProcessing file: {file_path}")
        print(f"Using {self.provider} with model: {self.model}")
        
        # Determine file type and extract text
        if file_path.suffix.lower() == '.pdf':
            if process_pdf is None:
                print("PDF support not available. Please ensure pdf_to_text module is available.")
                return None
            print("Extracting text from PDF...")
            raw_text = process_pdf(str(file_path))
            is_ocr = True
            extracted_metadata = metadata or {}
        elif file_path.suffix.lower() == '.epub':
            print("Extracting text from EPUB...")
            raw_text, extracted_metadata = self.epub_processor.extract_text_from_epub(str(file_path))
            is_ocr = False
            # Merge provided metadata with extracted metadata
            if metadata:
                extracted_metadata.update(metadata)
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return None
        
        if not raw_text:
            print("Failed to extract text from file.")
            return None
            
        # Set default output path if not provided
        if not output_path:
            output_path = file_path.stem + "_enhanced.txt"
            
        # Enhance text with AI
        print("\nEnhancing text with AI...")
        start_time = time.time()
        enhanced_text = self.enhance_text(raw_text, context=extracted_metadata, is_ocr=is_ocr)
        end_time = time.time()
        
        print(f"\nProcessing completed in {end_time - start_time:.1f} seconds")
        
        # Save enhanced text
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata as header
            f.write("=" * 50 + "\n")
            f.write(f"Title: {extracted_metadata.get('title', 'Unknown')}\n")
            f.write(f"Author: {extracted_metadata.get('author', 'Unknown')}\n")
            f.write(f"Subject: {extracted_metadata.get('subject', 'Unknown')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(enhanced_text)
            
        print(f"Enhanced text saved to: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Enhance text from PDFs and EPUBs with AI corrections')
    parser.add_argument('file_path', help='Path to the PDF or EPUB file')
    parser.add_argument('-o', '--output', help='Output path for enhanced text')
    parser.add_argument('--provider', default='ollama', choices=['ollama', 'openai'], 
                       help='AI provider to use (default: ollama)')
    parser.add_argument('--api-key', help='API key (only required for OpenAI)')
    parser.add_argument('--model', help='Model to use (default: mistral:latest for Ollama, gpt-4 for OpenAI)')
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
    try:
        enhancer = AiTextEnhancer(
            provider=args.provider,
            api_key=args.api_key,
            model=args.model,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        
        # Process file
        enhancer.process_file(args.file_path, args.output, metadata)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()