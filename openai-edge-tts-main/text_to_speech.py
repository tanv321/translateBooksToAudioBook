import os
import re
import json
import time
import requests
from pathlib import Path

# Configuration
API_URL = "http://localhost:5050/v1/audio/speech"
API_KEY = "7127321731731vbhx21737123SHSSAGDGADGVAS72E7EGUDQHUQWDYQDB"  # Your API key
VOICE = "en-US-SteffanNeural"  # Change to your preferred voice
RESPONSE_FORMAT = "mp3"
MAX_CHARS = 4000  # Slightly less than 4096 to be safe
OUTPUT_DIR = "audiobook_output"  # Directory for output files

def read_text_file(file_path):
    """Read the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def split_into_chunks(text, max_chars=MAX_CHARS):
    """Split text into chunks respecting sentence boundaries."""
    chunks = []
    current_chunk = ""
    
    # Split text into sentences
    # This regex matches sentences ending with ., !, or ? followed by a space or newline
    sentences = re.split(r'([.!?][\s\n])', text)
    
    # Recombine sentences with their ending punctuation
    complete_sentences = []
    i = 0
    while i < len(sentences) - 1:
        if i + 1 < len(sentences) and re.match(r'[.!?][\s\n]', sentences[i+1]):
            complete_sentences.append(sentences[i] + sentences[i+1])
            i += 2
        else:
            complete_sentences.append(sentences[i])
            i += 1
    
    # Add the last sentence if there's an odd number
    if i < len(sentences):
        complete_sentences.append(sentences[i])
    
    # Combine sentences into chunks
    for sentence in complete_sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            if current_chunk:  # Don't append empty chunks
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def text_to_speech(text, output_file, voice=VOICE, response_format=RESPONSE_FORMAT):
    """Convert text to speech using the Edge TTS API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "input": text,
        "voice": voice,
        "response_format": response_format
    }
    
    print(f"Sending request to API with {len(text)} characters")
    print(f"Output will be saved to: {os.path.abspath(output_file)}")
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        
        if response.status_code == 200:
            # Debug info about response
            content_length = len(response.content)
            print(f"Received response: {content_length} bytes")
            
            if content_length < 100:  # Suspiciously small file
                print(f"WARNING: Response content is very small ({content_length} bytes)")
                print(f"Response content: {response.content}")
            
            # Save the file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Verify file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"File saved successfully: {file_size} bytes")
                if file_size < 100:  # Suspiciously small file
                    print(f"WARNING: Saved file is very small ({file_size} bytes)")
            else:
                print(f"ERROR: File was not created at {output_file}")
                
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Request error: {e}")
        return False

def process_text_file(input_file, output_dir=OUTPUT_DIR, voice=VOICE):
    """Process an entire text file into multiple audio files."""
    # Create output directory if it doesn't exist
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir_path.absolute()}")
    
    # Read the text file
    text = read_text_file(input_file)
    if not text:
        print(f"ERROR: Failed to read text from {input_file}")
        return False
    
    print(f"Successfully read {len(text)} characters from {input_file}")
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Split into chunks
    chunks = split_into_chunks(text)
    print(f"Text split into {len(chunks)} chunks")
    
    # Show first few characters of the first chunk as a sample
    if chunks:
        sample = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
        print(f"Sample of first chunk: \"{sample}\"")
    
    # Process each chunk
    successful_chunks = 0
    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"{base_name}_part{i+1:03d}.{RESPONSE_FORMAT}")
        print(f"\nProcessing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        
        # Skip empty chunks
        if not chunk.strip():
            print(f"WARNING: Chunk {i+1} is empty, skipping")
            continue
            
        if text_to_speech(chunk, output_file, voice):
            successful_chunks += 1
            print(f"✓ Chunk {i+1} processed successfully")
        else:
            print(f"✗ Failed to process chunk {i+1}")
        
        # Check if file exists and is playable
        if os.path.exists(output_file):
            if os.path.getsize(output_file) < 1000:  # Less than 1KB is suspicious
                print(f"WARNING: Output file is suspiciously small: {os.path.getsize(output_file)} bytes")
        
        # Small delay to avoid overwhelming the API
        if i < len(chunks) - 1:
            time.sleep(0.5)
    
    print(f"\nCompleted! {successful_chunks}/{len(chunks)} chunks processed successfully.")
    
    # Print list of created files
    output_files = list(output_dir_path.glob(f"{base_name}_part*.{RESPONSE_FORMAT}"))
    print(f"\nCreated {len(output_files)} output files:")
    for file in sorted(output_files):
        print(f"  - {file.name} ({os.path.getsize(file)} bytes)")
    
    return successful_chunks == len(chunks)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert text file to speech using Edge TTS API")
    parser.add_argument("input_file", help="Input text file path")
    parser.add_argument("--voice", default=VOICE, help=f"Voice to use (default: {VOICE})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--format", default=RESPONSE_FORMAT, help=f"Audio format (default: {RESPONSE_FORMAT})")
    parser.add_argument("--test", action="store_true", help="Run a quick test with a simple sentence")
    
    args = parser.parse_args()
    
    # Update global variables with args
    VOICE = args.voice
    RESPONSE_FORMAT = args.format
    OUTPUT_DIR = args.output_dir
    
    # Print configuration
    print(f"\nCONFIGURATION:")
    print(f"API URL: {API_URL}")
    print(f"Voice: {VOICE}")
    print(f"Format: {RESPONSE_FORMAT}")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Working Directory: {os.getcwd()}\n")
    
    # Create a simple test file if needed
    if args.test:
        test_file = "quick_test.txt"
        test_text = "This is a quick test to verify that the text-to-speech functionality is working properly."
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_text)
        print(f"Created test file: {test_file}")
        
        # Process the test file
        test_output = "test_output.mp3"
        print(f"Running quick test to {test_output}...")
        text_to_speech(test_text, test_output)
        print(f"Quick test complete. Check if {os.path.abspath(test_output)} plays correctly.\n")
    
    # Process the main file if not just testing
    if not args.test or os.path.exists(args.input_file):
        # Verify input file exists
        if not os.path.exists(args.input_file):
            print(f"ERROR: Input file not found: {args.input_file}")
        else:
            print(f"Processing file: {os.path.abspath(args.input_file)}")
            process_text_file(args.input_file, args.output_dir, args.voice)