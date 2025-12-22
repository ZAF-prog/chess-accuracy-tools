import sys
from pathlib import Path
import PyPDF2

pdf_path = Path(r'C:/Users/Public/Github/chess-accuracy-tools/data/Kenneth Regan _ played through the positions in the Solitaire Set_convo.pdf')
if not pdf_path.is_file():
    print('PDF file not found:', pdf_path)
    sys.exit(1)

print(f"Opening PDF: {pdf_path}")
try:
    with pdf_path.open('rb') as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        print(f"Total pages: {num_pages}")
        text = ''
        for i, page in enumerate(reader.pages):
            print(f"Extracting page {i+1}/{num_pages}...", flush=True)
            txt = page.extract_text()
            if txt:
                text += txt + '\n'
except Exception as e:
    print(f"Error during extraction: {e}")
    sys.exit(1)

print("\nExtraction complete. First 2000 characters:\n")
print(text[:2000])

# Also save to a text file for review
output_txt = pdf_path.with_suffix('.txt')
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write(text)
print(f"\nFull text saved to: {output_txt}")
