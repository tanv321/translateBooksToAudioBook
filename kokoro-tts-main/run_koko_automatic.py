import os
import subprocess
import textwrap
import re
import shutil
from pathlib import Path
import zipfile
from bs4 import BeautifulSoup
import time

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

def sanitize_filename(filename):
    """Clean filename for safe directory/file creation."""
    # Remove file extension and clean the name
    name = Path(filename).stem
    # Replace problematic characters with underscores
    name = re.sub(r'[^\w\-_\.]', '_', name)
    # Replace multiple underscores with single ones
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def process_single_book(input_file, speed, lang, voice, kokoro_dir, audiobooks_dir):
    """Process a single book file."""
    input_path = Path(input_file)
    
    print(f"\n{'='*60}")
    print(f"📖 Processing: {input_path.name}")
    print(f"{'='*60}")
    
    # Determine file type and extract text
    if input_path.suffix.lower() == '.epub':
        print("Extracting text from EPUB file...")
        text_content = extract_text_from_epub(input_path)
        if text_content is None:
            print(f"❌ Failed to extract text from {input_path.name}")
            return False
        print(f"✅ Extracted {len(text_content):,} characters from EPUB.")
    elif input_path.suffix.lower() == '.txt':
        print("Reading text file...")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            print(f"✅ Read {len(text_content):,} characters from text file.")
        except Exception as e:
            print(f"❌ Failed to read {input_path.name}: {e}")
            return False
    else:
        print(f"⚠️ Unsupported file type: {input_path.suffix}")
        return False
    
    # Create book output directory
    book_name = sanitize_filename(input_path.name)
    book_output_dir = audiobooks_dir / book_name
    
    # Check if book is already processed
    if book_output_dir.exists():
        existing_wavs = list(book_output_dir.glob("*.wav"))
        if existing_wavs:
            print(f"📁 Book already processed ({len(existing_wavs)} files found) - SKIPPING")
            return True
    
    # Create directories
    book_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for chunk files
    temp_dir = kokoro_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {book_output_dir}")
    
    # Split the text into chunks
    chunks = split_text_into_chunks(text_content)
    print(f"📄 Split into {len(chunks)} chunks")
    
    # Change to kokoro directory to run the executable
    original_cwd = os.getcwd()
    os.chdir(kokoro_dir)
    
    start_time = time.time()
    
    try:
        for i, chunk in enumerate(chunks, 1):
            # Create temp chunk file
            chunk_file = temp_dir / f"chunk_{i}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            # Create output filename
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
            
            print(f"🎤 Processing chunk {i}/{len(chunks)}: {output_file.name}")
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"✅ Successfully processed chunk {i}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error processing chunk {i}: {e}")
                return False
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 Book completed in {elapsed_time/60:.1f} minutes!")
        
        # Show summary
        wav_files = list(book_output_dir.glob("*.wav"))
        total_size = sum(f.stat().st_size for f in wav_files) / (1024*1024)  # MB
        print(f"📊 Generated {len(wav_files)} audio files ({total_size:.1f} MB)")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Processing interrupted by user")
        return False
    finally:
        # Return to original directory
        os.chdir(original_cwd)
        
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def batch_process_books():
    """Main batch processing function."""
    print("=== Kokoro TTS Batch Audiobook Generator ===\n")
    
    # Set up paths
    books_dir = Path("/Users/tanval/Desktop/Programming projects/translatingBooksToAudioBooks/Books/History")
    kokoro_dir = Path("/Users/tanval/Desktop/Programming projects/translatingBooksToAudioBooks/kokoro-tts-main")
    audiobooks_dir = Path("/Users/tanval/Desktop/Programming projects/translatingBooksToAudioBooks/kokoro-tts-main/Audiobooks")
    
    # Verify paths exist
    if not books_dir.exists():
        print(f"❌ Books directory not found: {books_dir}")
        return
    
    if not kokoro_dir.exists():
        print(f"❌ Kokoro TTS directory not found: {kokoro_dir}")
        return
    
    if not (kokoro_dir / "kokoro-tts").exists():
        print(f"❌ Kokoro TTS executable not found in: {kokoro_dir}")
        return
    
    # Create audiobooks directory
    audiobooks_dir.mkdir(exist_ok=True)
    
    # Find all supported book files
    supported_files = []
    for pattern in ["*.txt", "*.epub"]:
        supported_files.extend(books_dir.glob(pattern))
    
    if not supported_files:
        print(f"❌ No supported files found in {books_dir}")
        return
    
    print(f"📚 Found {len(supported_files)} books to process:")
    for i, file in enumerate(supported_files, 1):
        file_type = file.suffix.upper()
        print(f"  {i}. {file.name} ({file_type})")
    
    # Check dependencies for EPUB files
    epub_files = [f for f in supported_files if f.suffix.lower() == '.epub']
    if epub_files:
        try:
            import bs4
        except ImportError:
            print("\n⚠️ EPUB support requires BeautifulSoup4.")
            print("Install it with: pip install beautifulsoup4")
            print("Optionally, for better EPUB support: pip install ebooklib")
            return
    
    # Get settings
    print(f"\n⚙️ Configuration:")
    speed = 0.8
    lang = "en-us" 
    voice = "am_echo"
    
    print(f"  📖 Books directory: {books_dir}")
    print(f"  🎵 Output directory: {audiobooks_dir}")
    print(f"  🔧 Kokoro directory: {kokoro_dir}")
    print(f"  ⚡ Speed: {speed}")
    print(f"  🗣️ Voice: {voice}")
    print(f"  🌍 Language: {lang}")
    
    print(f"\n🚀 Starting batch processing of all {len(supported_files)} books...")
    print("🤖 Running in autonomous mode - no confirmations needed!")
    
    # Process each book
    total_start_time = time.time()
    processed_count = 0
    failed_books = []
    
    for i, book_file in enumerate(supported_files, 1):
        print(f"\n🔄 Book {i}/{len(supported_files)}")
        
        success = process_single_book(
            book_file, speed, lang, voice, kokoro_dir, audiobooks_dir
        )
        
        if success:
            processed_count += 1
            print(f"✅ Book {i} completed successfully!")
        else:
            failed_books.append(book_file.name)
            print(f"❌ Book {i} failed - continuing to next book...")
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"🏁 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {processed_count}/{len(supported_files)} books")
    print(f"⏱️ Total time: {total_elapsed/3600:.1f} hours")
    print(f"📁 Output location: {audiobooks_dir}")
    
    if failed_books:
        print(f"❌ Failed books:")
        for book in failed_books:
            print(f"  - {book}")

def main():
    """Main function - direct to batch processing."""
    try:
        batch_process_books()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Processing stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()