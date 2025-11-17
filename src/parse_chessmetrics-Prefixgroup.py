import requests
from bs4 import BeautifulSoup
import re
import os

def extract_prefix_links_from_local_html(file_path: str):
    """
    Reads a local HTML file, analyzes the embedded JavaScript, and reconstructs 
    the full 'FindPlayerByPrefix.asp' URLs from the FindPrefixGroup function calls.
    """
    
    # 1. Base URL structure to prepend to relative links
    BASE_URL = "http://chessmetrics.com/cm/CM2/"
    
    # 2. Define the static part of the URL found in the JavaScript function
    STATIC_URL_PREFIX = "FindPlayerByPrefix.asp?Params=197701SSSSS5S1120531977121510000000000"
    STATIC_URL_SUFFIX = "10100"

    try:
        # Check if the file exists using the provided relative path
        if not os.path.exists(file_path):
            # If the file isn't found in the current working directory, print the full path being checked
            absolute_check_path = os.path.abspath(file_path)
            return f"Error: File not found. Checked path: {absolute_check_path}. Please ensure the script is run from the repo root directory."
        
        # Read the local HTML file content using 'windows-1252' encoding
        with open(file_path, 'r', encoding='windows-1252') as f:
            content = f.read()

    except Exception as e:
        return f"Error reading file: {e}"

    # 3. Parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')
    
    prefix_links = []
    
    # 4. Find all <a> tags that call the 'FindPrefixGroup' JavaScript function
    # Regex captures the number inside the function call: '(\d+)'
    js_regex = re.compile(r"javascript:FindPrefixGroup\('(\d+)'\);")
    
    # Find all links that match the JavaScript function call pattern
    all_links = soup.find_all('a', href=js_regex)
    
    # 5. Extract the parameter and reconstruct the URL
    for link in all_links:
        href = link['href']
        link_text = link.get_text(strip=True)
        
        # Use the regex to find the prefix number (e.g., '363')
        match = js_regex.match(href)
        
        if match:
            prefix_group_id = match.group(1)
            
            # Reconstruct the full relative URL using the static parts and the dynamic ID
            relative_url = f"{STATIC_URL_PREFIX}{prefix_group_id}{STATIC_URL_SUFFIX}"
            
            # Create the absolute URL to the Chessmetrics site
            full_link = BASE_URL + relative_url
            
            prefix_links.append({
                'prefix': link_text,
                'prefix_id': prefix_group_id,
                'link': full_link
            })

    # 6. Output Results
    if prefix_links:
        print(f"Successfully reconstructed {len(prefix_links)} 'prefixgroup' links:")
        print("----------------------------------------------------------------------")
        for item in prefix_links:
            # We use repr() on the prefix key to show that the names are correctly read (e.g., 'Zy')
            print(f"Prefix: {repr(item['prefix']):<4} | ID: {item['prefix_id']:<3} | Link: {item['link']}")
        print("----------------------------------------------------------------------")
    else:
        print("Could not find any links calling FindPrefixGroup().")
        
    return prefix_links

# --- Usage ---
# The path is now relative to the Github repo root 'C:\Users\Public\Github\chess-accuracy-tools\'
# You MUST run your script from this root directory for the path to resolve correctly.
# The original file path was: 'C:\Users\Public\Github\chess-accuracy-tools\data\chessmetrics _ Find A Player.htm'
file_path = os.path.join('data', 'chessmetrics _ Find A Player.htm') 

extracted_data = extract_prefix_links_from_local_html(file_path)