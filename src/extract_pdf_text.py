#!/usr/bin/env python
import sys
import argparse
from pathlib import Path

def try_pypdf2(path):
    try:
        import PyPDF2
        print(f"PyPDF2 analyzing: {path}")
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            print(f"Number of pages: {len(reader.pages)}")
            
            # Extract text from pages 7-10
            text = []
            for i in range(1, min(10, len(reader.pages))):
                page = reader.pages[i]
                content = page.extract_text()
                text.append(f"\n--- PAGE {i+1} ---\n{content}")
            
            input_path = Path(path)
            output_file = input_path.with_name(f"{input_path.stem}_PDF2.txt")
            with open(output_file, "w", encoding="utf-8") as out:
                out.write("\n".join(text))
            return f"Wrote to {output_file}"
                
    except ImportError:
        try:
            import pypdf
            print(f"pypdf analyzing: {path}")
            with open(path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                print(f"Number of pages: {len(reader.pages)}")
                
                text = []
                for i in range(1, min(10, len(reader.pages))):
                    page = reader.pages[i]
                    content = page.extract_text()
                    # print(f"\n--- PAGE {i+1} ---")
                    # print(content)
                    text.append(f"\n--- PAGE {i+1} ---\n{content}")
                
                input_path = Path(path)
                output_file = input_path.with_name(f"{input_path.stem}_PDF2.txt")
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write("\n".join(text))
                return f"Wrote to {output_file}"
        except ImportError:
            print("PyPDF2/pypdf missing, trying pdfplumber...")
            try_pdfplumber(path)
    except Exception as e:
        print(f"PyPDF2 Error: {e}")

def try_pdfplumber(path):
    try:
        import pdfplumber
        print(f"pdfplumber analyzing: {path}")
        with pdfplumber.open(path) as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            
            text = []
            for i in range(1, min(10, len(pdf.pages))):
                page = pdf.pages[i]
                content = page.extract_text()
                # print(f"\n--- PAGE {i+1} ---") 
                # print(content)
                text.append(f"\n--- PAGE {i+1} ---\n{content}")
            
            input_path = Path(path)
            output_file = input_path.with_name(f"{input_path.stem}_PDF2.txt")
            with open(output_file, "w", encoding="utf-8") as out:
                out.write("\n".join(text))
            print(f"Wrote to {output_file} using pdfplumber")
    except ImportError:
        print("Neither PyPDF2 nor pdfplumber found.")
    except Exception as e:
        print(f"pdfplumber Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = r'C:\Users\Public\Github\chess-accuracy-tools\data\Kenneth Regan _ Intrinsic Ratings Compendium.pdf'
    
    try_pypdf2(target_path)
