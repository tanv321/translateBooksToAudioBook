import os
import subprocess
import textwrap
import re
import shutil
from pathlib import Path
import zipfile
from bs4 import BeautifulSoup

def extract_text_from_epub(epub_path):
    """Extract text content from EPUB file."""
    try:
        # Try using ebooklib if available
        import ebooklib
        from ebooklib import epub
        
        book = epub.read_epub(epub_path)
        text_content = []
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text_content.append(soup.get_text())
        
        return '\n\n'.join(text_content)
        
    except ImportError:
        print("ebooklib not found. Trying manual extraction...")
        return extract_text_from_epub_manual(epub_path)

def extract_text_from_epub_manual(epub_path):
    """Manually extract text from EPUB using zipfile and BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("BeautifulSoup4 is required for EPUB processing.")
        print("Install it with: pip install beautifulsoup4")
        return None
    
    text_content = []
    
    with zipfile.ZipFile(epub_path, 'r') as zip_file:
        # Get all HTML/XHTML files
        for file_info in zip_file.filelist:
            if file_info.filename.endswith(('.html', '.xhtml', '.htm')):
                try:
                    content = zip_file.read(file_info.filename).decode('utf-8')
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Extract text and clean it up
                    text = soup.get_text()
                    # Clean up extra whitespace
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                    text = re.sub(r' +', ' ', text)
                    
                    if text.strip():
                        text_content.append(text.strip())
                        
                except Exception as e:
                    print(f"Warning: Could not process {file_info.filename}: {e}")
    
    return '\n\n'.join(text_content)

def split_text_into_chunks(text, max_chars=4000):
    """Split text into chunks, respecting paragraph and sentence boundaries."""
    if isinstance(text, str):
        # Text is already loaded
        pass
    else:
        # Text is a file path
        with open(text, 'r', encoding='utf-8') as f:
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

def process_file_with_kokoro_tts(input_file, speed, lang, voice, kokoro_dir=None):
    """Process text or EPUB file in chunks using kokoro-tts with organized output."""
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Determine file type and extract text
    if input_path.suffix.lower() == '.epub':
        print("Extracting text from EPUB file...")
        text_content = extract_text_from_epub(input_path)
        if text_content is None:
            print("Failed to extract text from EPUB file.")
            return
        print(f"Extracted {len(text_content)} characters from EPUB.")
    else:
        # Assume it's a text file
        with open(input_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    
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
    chunks = split_text_into_chunks(text_content)
    
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
    print("=== Kokoro TTS Audiobook Generator (TXT & EPUB Support) ===\n")
    
    # Get current directory and look for supported files
    current_dir = Path.cwd()
    supported_files = list(current_dir.glob("*.txt")) + list(current_dir.glob("*.epub"))
    
    if supported_files:
        print("Found supported files in current directory:")
        for i, file in enumerate(supported_files, 1):
            file_type = "EPUB" if file.suffix.lower() == '.epub' else "TXT"
            print(f"  {i}. {file.name} ({file_type})")
        
        try:
            choice = int(input(f"\nSelect a file (1-{len(supported_files)}) or 0 for custom path: "))
            if 1 <= choice <= len(supported_files):
                input_file = supported_files[choice - 1]
            else:
                input_file = Path(input("Enter full path to text/EPUB file: "))
        except ValueError:
            input_file = Path(input("Enter full path to text/EPUB file: "))
    else:
        input_file = Path(input("Enter full path to text/EPUB file: "))
    
    # Check if we need to install dependencies for EPUB
    if input_file.suffix.lower() == '.epub':
        try:
            import bs4
        except ImportError:
            print("\n⚠️  EPUB support requires BeautifulSoup4.")
            print("Install it with: pip install beautifulsoup4")
            print("Optionally, for better EPUB support: pip install ebooklib")
            return
    
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
    
    process_file_with_kokoro_tts(
        str(input_file), 
        speed, 
        lang, 
        voice, 
        kokoro_dir
    )

if __name__ == "__main__":
    main()