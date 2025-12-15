import sys

def extract_ipr_info():
    try:
        import pypdf
        print("pypdf is available")
        path = r'C:\Users\Public\Github\chess-accuracy-tools\data\Kenneth Regan _ Intrinsic Ratings Compendium.pdf'
        reader = pypdf.PdfReader(path)
        print(f"Number of pages: {len(reader.pages)}")
        
        # Scan pages for key terms
        key_terms = ["IPR", "Intrinsic Performance Rating", "formula", "calculate", "parameter", "sensitivity", "consistency"]
        
        extracted_text = ""
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            # Simple heuristic to keep relevant pages
            if any(term in text for term in key_terms):
                extracted_text += f"\n\n--- PAGE {i+1} ---\n{text}"
                
        # Save to a text file for analysis
        output_path = r'C:\Users\Public\Github\chess-accuracy-tools\data\regan_pdf_content.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
            
        print(f"Extraction complete. Text saved to {output_path}")
        print("First 1000 characters:")
        print(extracted_text[:1000])

    except ImportError:
        print("pypdf not installed properly.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_ipr_info()
