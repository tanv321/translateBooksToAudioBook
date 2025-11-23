import os
import re
import shutil
from pathlib import Path
import zipfile
import subprocess
import time

def clean_text_for_tts(text):
    """Clean text for TTS: remove footnotes, section numbers, extra spacing."""
    # Remove superscript numbers and footnote markers
    text = re.sub(r'[\u00b9\u00b2\u00b3\u2074-\u2089]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\{\s*\d+\s*\}', '', text)
    text = re.sub(r'\s+\(\s*\d+\s*\)\s+', ' ', text)
    
    # Remove hierarchical section numbers (3.4.2.2.1)
    text = re.sub(r'^\s*\d+(?:\.\d+)+\s*', '', text, flags=re.MULTILINE)
    
    # Fix quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', '...')
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix excessive dots and dashes
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'-{2,}', '-', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    
    # Handle line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Remove control characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t ')
    
    return text.strip()

def extract_text_from_epub(path):
    """Extract text from EPUB."""
    try:
        with zipfile.ZipFile(path, 'r') as z:
            from bs4 import BeautifulSoup
            text = []
            for f in z.namelist():
                if f.endswith(('.html', '.xhtml', '.htm')):
                    content = z.read(f).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(content, 'html.parser')
                    for tag in soup(['script', 'style']):
                        tag.decompose()
                    if soup.get_text().strip():
                        text.append(soup.get_text())
            return '\n\n'.join(text)
    except Exception as e:
        print(f"❌ EPUB extraction failed: {e}")
        return None

def extract_text_from_pdf(path):
    """Extract text from PDF."""
    try:
        import fitz
        doc = fitz.open(path)
        text = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        return '\n\n'.join(text)
    except ImportError:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return '\n\n'.join(p.extract_text() or '' for p in pdf.pages)
        except ImportError:
            print("❌ PDF needs: pip install PyMuPDF or pip install pdfplumber")
            return None

def extract_text_from_mobi(path):
    """Extract text from MOBI (via Calibre conversion)."""
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            epub = Path(temp_dir) / "temp.epub"
            subprocess.run(["ebook-convert", str(path), str(epub)], check=True, capture_output=True)
            return extract_text_from_epub(epub)
    except Exception as e:
        print(f"❌ MOBI extraction failed: {e}")
        return None

def extract_text(file_path):
    """Extract text from any supported format."""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.epub':
        return extract_text_from_epub(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.mobi':
        return extract_text_from_mobi(file_path)
    else:
        print(f"❌ Unsupported format: {ext}")
        return None

def split_into_chunks(text, max_chars=4000):
    """Split text into chunks by paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""
    
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para + "\n\n"
        else:
            current += para + "\n\n"
    
    if current:
        chunks.append(current.strip())
    return chunks

def standardize_book(input_file, output_dir=None):
    """Extract, clean, and save a standardized text file."""
    input_path = Path(input_file)
    print(f"\n📖 Extracting: {input_path.name}")
    
    text = extract_text(input_file)
    if not text:
        return None
    
    print(f"✅ Extracted {len(text):,} characters")
    print(f"🧹 Cleaning text...")
    
    text = clean_text_for_tts(text)
    print(f"✅ Cleaned to {len(text):,} characters")
    
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{input_path.stem}_standardized.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"✅ Saved: {output_file}")
    return output_file

def process_to_audio(book_file, speed, lang, voice, kokoro_dir, audiobooks_dir):
    """Process standardized book to audio."""
    print(f"\n🎵 Processing: {Path(book_file).name}")
    
    with open(book_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    book_name = Path(book_file).stem.replace('_standardized', '')
    output_dir = audiobooks_dir / book_name
    
    if output_dir.exists() and list(output_dir.glob("*.wav")):
        print(f"⏭️  Already processed - skipping")
        return True
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = kokoro_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    chunks = split_into_chunks(text)
    print(f"📄 Split into {len(chunks)} chunks")
    
    original_cwd = os.getcwd()
    os.chdir(kokoro_dir)
    start = time.time()
    
    try:
        for i, chunk in enumerate(chunks, 1):
            chunk_file = temp_dir / f"chunk_{i}.txt"
            chunk_file.write_text(chunk, encoding='utf-8')
            
            output_file = output_dir / f"{book_name}_{i:03d}.wav"
            
            cmd = ["./kokoro-tts", str(chunk_file), str(output_file),
                   "--speed", str(speed), "--lang", lang, "--voice", voice]
            
            print(f"🎤 Chunk {i}/{len(chunks)}", end='\r')
            subprocess.run(cmd, check=True, capture_output=True)
        
        elapsed = (time.time() - start) / 60
        wav_count = len(list(output_dir.glob("*.wav")))
        size_mb = sum(f.stat().st_size for f in output_dir.glob("*.wav")) / (1024*1024)
        
        print(f"\n🎉 Done! {wav_count} files, {size_mb:.1f} MB in {elapsed:.1f} min")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        os.chdir(original_cwd)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    print("=== TTS Book Processor ===\n")
    print("1. Standardize book (extract + clean)")
    print("2. Process to audio (standardized → TTS)")
    print("3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        file = input("Book path: ").strip()
        output = input("Output dir (or Enter for current): ").strip() or None
        standardize_book(file, output)
        
    elif choice == '2':
        books_dir = Path("/path/to/your/books")  # EDIT THIS
        kokoro_dir = Path("/path/to/kokoro")      # EDIT THIS
        audio_dir = kokoro_dir / "Audiobooks"
        
        if not books_dir.exists():
            print(f"❌ Edit book paths in script")
            return
        
        audio_dir.mkdir(exist_ok=True)
        books = list(books_dir.glob("*.txt")) + list(books_dir.glob("*.epub")) + \
                list(books_dir.glob("*.pdf")) + list(books_dir.glob("*.mobi"))
        
        if not books:
            print(f"❌ No books found")
            return
        
        print(f"📚 Found {len(books)} books\n")
        
        for book in books:
            process_to_audio(book, 0.8, "en-us", "am_echo", kokoro_dir, audio_dir)

if __name__ == "__main__":
    main()