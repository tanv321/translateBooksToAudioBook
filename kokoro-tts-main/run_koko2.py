import os
import subprocess
import textwrap
import re
import shutil
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

def find_kokoro_tts_directory():
    """Find the kokoro-tts-main directory by searching from current location."""
    current_path = Path.cwd()
    
    # Search up the directory tree for kokoro-tts-main
    for parent in [current_path] + list(current_path.parents):
        kokoro_path = parent / "kokoro-tts-main"
        if kokoro_path.exists() and (kokoro_path / "kokoro-tts").exists():
            return kokoro_path
    
    # If not found, ask user for the path
    print("Could not automatically find kokoro-tts-main directory.")
    kokoro_path = input("Please enter the full path to kokoro-tts-main directory: ")
    return Path(kokoro_path)

def process_text_with_kokoro_tts(input_file, speed, lang, voice, kokoro_dir=None):
    """Process text in chunks using kokoro-tts with organized output."""
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Find kokoro-tts directory
    if kokoro_dir is None:
        kokoro_dir = find_kokoro_tts_directory()
    else:
        kokoro_dir = Path(kokoro_dir)
    
    print(f"Using kokoro-tts directory: {kokoro_dir}")
    
    # Create audiobooks directory structure
    audiobooks_dir = kokoro_dir / "Audiobooks"
    book_name = input_path.stem  # filename without extension
    book_output_dir = audiobooks_dir / book_name
    
    # Create directories
    book_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for chunk files in the kokoro directory
    temp_dir = kokoro_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {book_output_dir}")
    
    # Split the text into chunks
    chunks = split_text_into_chunks(input_file)
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Change to kokoro directory to run the executable
    original_cwd = os.getcwd()
    os.chdir(kokoro_dir)
    
    try:
        for i, chunk in enumerate(chunks, 1):
            # Create temp chunk file
            chunk_file = temp_dir / f"chunk_{i}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Create output filename in the organized directory
            output_file = book_output_dir / f"{book_name}_{i:03d}.wav"
            
            # Run kokoro-tts command
            cmd = [
                "./kokoro-tts",
                str(chunk_file),
                str(output_file),
                "--speed", str(speed),
                "--lang", lang,
                "--voice", voice
            ]
            
            print(f"Processing chunk {i}/{len(chunks)}: {output_file.name}")
            try:
                subprocess.run(cmd, check=True)
                print(f"✓ Successfully processed chunk {i}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error processing chunk {i}: {e}")
        
        print(f"\n🎉 All chunks processed!")
        print(f"📁 Audio files saved to: {book_output_dir}")
        
        # Show summary
        wav_files = list(book_output_dir.glob("*.wav"))
        print(f"📊 Generated {len(wav_files)} audio files")
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)
        
        # Clean up temporary files
        cleanup = input("\nDo you want to remove temporary chunk files? (y/n): ")
        if cleanup.lower() == 'y':
            shutil.rmtree(temp_dir)
            print("🧹 Temporary files removed.")

def main():
    """Main function to run the script."""
    print("=== Kokoro TTS Audiobook Generator ===\n")
    
    # Get current directory and look for text files
    current_dir = Path.cwd()
    txt_files = list(current_dir.glob("*.txt"))
    
    if txt_files:
        print("Found text files in current directory:")
        for i, file in enumerate(txt_files, 1):
            print(f"  {i}. {file.name}")
        
        try:
            choice = int(input(f"\nSelect a file (1-{len(txt_files)}) or 0 for custom path: "))
            if 1 <= choice <= len(txt_files):
                input_file = txt_files[choice - 1]
            else:
                input_file = Path(input("Enter full path to text file: "))
        except ValueError:
            input_file = Path(input("Enter full path to text file: "))
    else:
        input_file = Path(input("Enter full path to text file: "))
    
    # Default values
    speed = 0.8
    lang = "en-us"
    voice = "am_echo"
    
    # Get parameters from user
    print(f"\nDefault settings:")
    print(f"  Speed: {speed}")
    print(f"  Language: {lang}")
    print(f"  Voice: {voice}")
    
    use_defaults = input("\nUse default settings? (y/n): ")
    
    if use_defaults.lower() != 'y':
        try:
            speed = float(input(f"Speed (default: {speed}): ") or speed)
        except ValueError:
            print("Invalid speed, using default")
        
        lang = input(f"Language (default: {lang}): ") or lang
        voice = input(f"Voice (default: {voice}): ") or voice
    
    # Optional: specify kokoro directory manually
    kokoro_dir = input("Kokoro-tts directory (leave empty for auto-detect): ") or None
    
    print(f"\n🚀 Starting processing...")
    print(f"📖 Input: {input_file}")
    print(f"⚡ Speed: {speed}")
    print(f"🗣️  Voice: {voice}")
    print(f"🌍 Language: {lang}\n")
    
    process_text_with_kokoro_tts(
        str(input_file), 
        speed, 
        lang, 
        voice, 
        kokoro_dir
    )

if __name__ == "__main__":
    main()