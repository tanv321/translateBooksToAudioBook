import os
import subprocess
import textwrap
import re
from pathlib import Path

def split_text_into_chunks(input_file, max_chars=4000):
    """Split text into chunks, respecting paragraph and sentence boundaries."""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_chars and we already have content,
        # start a new chunk
        if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text_with_kokoro_tts(input_file, base_output_name, speed, lang, voice):
    """Process text in chunks using kokoro-tts."""
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(input_file))
    
    # Create a temporary directory for chunk files
    temp_dir = os.path.join(input_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split the text into chunks
    chunks = split_text_into_chunks(input_file)
    
    print(f"Processing {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks, 1):
        # Create temp chunk file
        chunk_file = os.path.join(temp_dir, f"chunk_{i}.txt")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(chunk)
        
        # Create output filename
        output_file = f"{base_output_name}_{i}.wav"
        
        # Run kokoro-tts command
        cmd = [
            "./kokoro-tts",
            chunk_file,
            output_file,
            "--speed", str(speed),
            "--lang", lang,
            "--voice", voice
        ]
        
        print(f"Processing chunk {i}/{len(chunks)}: {output_file}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed chunk {i}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing chunk {i}: {e}")
    
    print("All chunks processed!")
    
    # Option to clean up temporary files
    cleanup = input("Do you want to remove temporary chunk files? (y/n): ")
    if cleanup.lower() == 'y':
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        print("Temporary files removed.")

if __name__ == "__main__":
    # Get current directory
    current_dir = os.getcwd()
    
    # Default values based on your example
    input_file = os.path.join(current_dir, "enhanced_deliverance.txt")
    base_output_name = "enhanced_deliverance"
    speed = 0.8
    lang = "en-us"
    voice = "am_echo"
    
    # Optionally, get parameters from user input
    use_defaults = input("Use default parameters? (y/n): ")
    
    if use_defaults.lower() != 'y':
        input_file = input(f"Input file path (default: {input_file}): ") or input_file
        base_output_name = input(f"Base output name (default: {base_output_name}): ") or base_output_name
        speed = float(input(f"Speed (default: {speed}): ") or speed)
        lang = input(f"Language (default: {lang}): ") or lang
        voice = input(f"Voice (default: {voice}): ") or voice
    
    process_text_with_kokoro_tts(input_file, base_output_name, speed, lang, voice)