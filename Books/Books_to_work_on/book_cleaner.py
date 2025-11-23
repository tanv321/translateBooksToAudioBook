#!/usr/bin/env python3
"""
Book Text Cleaner for Audiobook Conversion
Automatically removes footnotes, citations, prefaces, and non-narrative content
Uses local Ollama (Llama 3.2) - completely free!
"""

import requests
import json
import sys
import os
import re
from pathlib import Path

def clean_chunk(text, model="llama3.2"):
    """Send a text chunk to local Ollama for cleaning"""
    
    prompt = f"""You are a text cleaning assistant for audiobook preparation. 

Remove ALL of the following:
- Footnote numbers and footnote content (e.g., "1. Author Name, Title...")
- Citations and references (e.g., "ibid., para. 91", "See Chapter X", "pp. 123-45")
- Page numbers
- Preface, foreword, acknowledgments, table of contents, index sections
- Image descriptions (e.g., "[Image: ...]", "Figure 1:", "Photo credit:")
- Academic formatting like "(2016)", "See Conclusion"
- Cross-references to other chapters or sections
- Copyright notices, ISBN numbers, publication info
- Author bio sections
- Any metadata or non-narrative content

Keep ONLY the main narrative/chapter content meant to be read aloud.

Output ONLY the cleaned text with no explanations. If the entire chunk is just footnotes/citations/metadata, output "REMOVE_CHUNK".

Text to clean:
{text}

Cleaned text:"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                }
            },
            timeout=180
        )
        
        if response.status_code == 200:
            cleaned = response.json()['response'].strip()
            return None if cleaned == "REMOVE_CHUNK" else cleaned
        else:
            print(f"\nError: Got status code {response.status_code}")
            return text
            
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama. Make sure it's running!")
        print("Ollama should already be running. If not, open a new terminal and run: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"\nError processing chunk: {e}")
        return text

def split_by_sentences(text, sentences_per_chunk=5):
    """
    Split text into chunks based on sentences (periods).
    This ensures we never cut mid-sentence or mid-paragraph.
    """
    # Split by periods followed by space and capital letter (end of sentence)
    # Also handle other sentence endings like ? and !
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    chunks = []
    current_chunk = []
    
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        
        # Create chunk every N sentences or at the end
        if (i + 1) % sentences_per_chunk == 0 or i == len(sentences) - 1:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    return chunks

def clean_book(input_file, output_file=None, sentences_per_chunk=5, show_preview=True):
    """Clean an entire book file"""
    
    # Read input file
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        sys.exit(1)
    
    print(f"📖 Reading '{input_file}'...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"✓ Book loaded: {len(text):,} characters\n")
    
    # Split into sentence-based chunks
    print(f"✂️  Splitting by sentences ({sentences_per_chunk} sentences per chunk)...")
    chunks = split_by_sentences(text, sentences_per_chunk)
    print(f"✓ Split into {len(chunks)} chunks\n")
    
    print("=" * 80)
    print("🧹 Starting cleaning process...")
    print("=" * 80)
    print()
    
    # Process each chunk
    cleaned_chunks = []
    removed_count = 0
    
    for i, chunk in enumerate(chunks, 1):
        # Show progress
        print(f"\n[Chunk {i}/{len(chunks)}]")
        
        # Show preview of what's being processed
        if show_preview:
            preview = chunk[:150].replace('\n', ' ')
            if len(chunk) > 150:
                preview += "..."
            print(f"📄 Processing: {preview}")
        
        # Clean the chunk
        cleaned = clean_chunk(chunk)
        
        if cleaned is None:
            print("❌ Removed (footnotes/citations only)")
            removed_count += 1
        else:
            cleaned_chunks.append(cleaned)
            # Show what came back
            if show_preview:
                cleaned_preview = cleaned[:150].replace('\n', ' ')
                if len(cleaned) > 150:
                    cleaned_preview += "..."
                print(f"✓ Cleaned: {cleaned_preview}")
    
    # Combine cleaned chunks
    cleaned_text = '\n\n'.join(cleaned_chunks)
    
    # Determine output filename
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.stem + '_cleaned' + input_path.suffix
    
    # Save cleaned text
    print("\n" + "=" * 80)
    print(f"💾 Saving cleaned text to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)
    print(f"📊 Original:        {len(text):,} characters")
    print(f"📊 Cleaned:         {len(cleaned_text):,} characters")
    print(f"📊 Removed:         {len(text) - len(cleaned_text):,} characters")
    print(f"📊 Chunks removed:  {removed_count}/{len(chunks)}")
    print(f"📁 Output file:     {output_file}")
    print("=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python book_cleaner.py <input_file> [sentences_per_chunk] [output_file]")
        print("\nExamples:")
        print("  python book_cleaner.py my_book.txt")
        print("  python book_cleaner.py my_book.txt 5")
        print("  python book_cleaner.py my_book.txt 5 clean_book.txt")
        print("\nDefault: 5 sentences per chunk")
        sys.exit(1)
    
    input_file = sys.argv[1]
    sentences_per_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("\n" + "=" * 80)
    print("📚 BOOK TEXT CLEANER FOR AUDIOBOOK CONVERSION")
    print("🤖 Using local Llama 3.2 via Ollama")
    print("=" * 80)
    print()
    
    clean_book(input_file, output_file, sentences_per_chunk)

if __name__ == "__main__":
    main()