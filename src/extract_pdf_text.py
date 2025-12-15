import sys

def try_pypdf2():
    try:
        import PyPDF2
        print("PyPDF2 is available")
        path = r'C:\Users\Public\Github\chess-accuracy-tools\data\Kenneth Regan _ Intrinsic Ratings Compendium.pdf'
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            print(f"Number of pages: {len(reader.pages)}")
            
            # Extract text from first 5 pages to get an overview
            for i in range(min(5, len(reader.pages))):
                print(f"\n--- PAGE {i+1} TEXT ---")
                print(reader.pages[i].extract_text())
                
    except ImportError:
        print("PyPDF2 not looking good, trying pdfplumber...")
        try_pdfplumber()
    except Exception as e:
        print(f"PyPDF2 Error: {e}")

def try_pdfplumber():
    try:
        import pdfplumber
        print("pdfplumber is available")
        path = r'C:\Users\Public\Github\chess-accuracy-tools\data\Kenneth Regan _ Intrinsic Ratings Compendium.pdf'
        with pdfplumber.open(path) as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            for i in range(min(5, len(pdf.pages))):
                print(f"\n--- PAGE {i+1} TEXT ---")
                print(pdf.pages[i].extract_text())
    except ImportError:
        print("Neither PyPDF2 nor pdfplumber found.")
    except Exception as e:
        print(f"pdfplumber Error: {e}")

if __name__ == "__main__":
    try_pypdf2()
