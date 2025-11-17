import requests
from bs4 import BeautifulSoup
import csv
import re
import os
import io
import time
import random
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# --- Global Constants ---
BASE_URL = "http://chessmetrics.com/cm/CM2/"
TEMP_OUTPUT_FILENAME = os.path.join('data', 'chessmetrics-monthly_top5_temp.csv') 
DEFAULT_START_FILE = os.path.join('data', 'chessmetrics _ Monthly_1843-01.htm')
DEFAULT_MAX_MONTHS = 99
# Chessmetrics uses Windows-1252 encoding
ENCODING = 'windows-1252' 

# --- Utility Functions ---

def throttle():
    """Introduces a random delay between 1.1 and 2.9 seconds to be polite to the server."""
    delay = random.uniform(1.1, 2.9)
    time.sleep(delay)

def format_player_name(full_name: str) -> str:
    """
    Converts 'Firstname(s) Lastname(s)' to 'Lastname(s), Firstname(s)' 
    and removes all periods from the resulting string.
    """
    parts = full_name.strip().split()
    if len(parts) > 1:
        # Assuming the last word is the last name
        last_name = parts[-1]
        first_names = ' '.join(parts[:-1])
        result = f"{last_name}, {first_names}"
    else:
        result = full_name
    
    return result.replace('.', '').strip()

def fetch_content(source: str) -> Optional[BeautifulSoup]:
    """
    Fetches content from a URL or loads a local file.
    Uses the correct Windows-1252 encoding.
    """
    if os.path.exists(source):
        # Local file
        print(f"Loading local file: {source}")
        try:
            with open(source, 'r', encoding=ENCODING) as f:
                content = f.read()
            return BeautifulSoup(content, 'html.parser') 
        except Exception as e:
            print(f"Error loading local file {source}: {e}")
            return None
            
    elif source.startswith('http'):
        # Network request
        throttle() 
        try:
            response = requests.get(source)
            response.raise_for_status()
            # Decode using the specified encoding
            response.encoding = ENCODING
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {source}: {e}")
            return None
    else:
        print(f"Invalid content source: {source}")
        return None

def get_last_url_from_csv(filepath: str) -> Optional[str]:
    """
    Reads the specified CSV file and extracts the URL from the last record (2nd field, index 1).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found at path: {filepath}")

    last_data_line = None
    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip header line
        next(f, None)
        
        # Iterate over lines to find the last non-empty one
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                last_data_line = stripped_line
    
    if not last_data_line:
        return None

    # Use io.StringIO and csv.reader to safely parse the CSV line
    reader = csv.reader(io.StringIO(last_data_line))
    try:
        row = next(reader)
        # The URL is the 2nd field (index 1)
        if len(row) > 1:
            return row[1].strip()
    except StopIteration:
        pass
    
    return None

def get_next_month_url(current_url: str) -> Optional[str]:
    """
    Parses the current SingleMonth URL and increments the YYYYMM part to get the next month's URL.
    
    FIX: Uses robust slicing based on fixed parameter lengths (6-digit ID, 6-digit Date, 21-digit Suffix).
    """
    # Regex targets the entire sequence of digits after the last 'S' in the parameter.
    match_sequence = re.search(r'S(\d+)', current_url)
    
    if not match_sequence:
        print("Warning: Could not find numeric parameter sequence in URL.")
        return None

    full_param_sequence = match_sequence.group(1) # e.g., '113454184301151000005300000210100'

    # The sequence is known to be: [Player ID (6)] + [YYYYMM (6)] + [Suffix (21)]
    if len(full_param_sequence) != 33:
        print(f"Error: Parameter sequence length is unexpected ({len(full_param_sequence)} instead of 33). Cannot increment date.")
        return None

    # Slice the fixed components
    player_id = full_param_sequence[0:6]
    current_yyyymm = full_param_sequence[6:12]
    suffix = full_param_sequence[12:] # Should be the remaining 21 digits

    try:
        current_date = datetime.strptime(current_yyyymm, '%Y%m')
        
        # Calculate next month
        if current_date.month == 12:
            next_date = datetime(current_date.year + 1, 1, 1)
        else:
            next_date = datetime(current_date.year, current_date.month + 1, 1)

        next_yyyymm = next_date.strftime('%Y%m')
        
        # Reconstruct the full parameter sequence
        next_param_sequence = player_id + next_yyyymm + suffix

        # Reconstruct the full URL
        prefix_start = match_sequence.start(1)
        # Prefix is the part of the URL before the full number sequence starts
        prefix = current_url[:prefix_start] 
        
        next_url = f"{prefix}{next_param_sequence}"
        return next_url
        
    except ValueError as e:
        print(f"Error processing date for URL increment: {e}")
        return None

# --- Core Scraping Function ---

def scrape_monthly_list(soup: BeautifulSoup, current_url: str) -> Optional[Tuple[List[Dict[str, Any]], str, str]]:
    """
    Extracts the date, top 5 player data, player profile template, and next month's URL.
    """
    
    # 1. Get Date from Title
    date_match = re.search(r'Monthly List: ([A-Za-z]+ \d{4}) rating list', soup.title.string if soup.title else "")
    if not date_match:
        print("Error: Could not extract date from page title.")
        return None
    
    date_str_long = date_match.group(1)
    date_obj = datetime.strptime(date_str_long, '%B %Y')
    yyyymm_date = date_obj.strftime('%Y-%m')
    
    # 2. Extract Player Profile URL Template
    player_url_template_suffix = ""
    js_code_tag = soup.find('script', string=lambda t: t and 'function LinkPlayer' in t)
    if js_code_tag and js_code_tag.string:
        # Match the suffix after the NewPlayerID variable insertion point
        url_match = re.search(
            r'var NewURL = "PlayerProfile\.asp\?Params=[^"]*"\s*\+\s*NewPlayerID\s*\+\s*"([^"]*)";', 
            js_code_tag.string, 
            re.DOTALL
        )
        if url_match:
            player_url_template_suffix = url_match.group(1)
    
    if not player_url_template_suffix:
        # Fallback to hardcoded suffix if JS parsing fails, to allow scraping to continue
        # This suffix comes from the uploaded HTML's LinkPlayer function.
        player_url_template_suffix = "184301151000005300000210100" 
        print(f"Warning: Falling back to hardcoded PlayerProfile suffix for {yyyymm_date}.")

    player_url_template_prefix = "PlayerProfile.asp?Params=184010SSSSS5S"

    # 3. Extract Top 5 Player Data
    player_data: List[Dict[str, Any]] = []
    
    for i in range(1, 6):
        rank_text = f"#{i}"
        
        # Find the specific font tag with the rank number
        rank_font = soup.find('font', string=lambda t: t and rank_text == t.strip())
        
        if not rank_font: 
            break 
            
        row = rank_font.find_parent('tr')
        if not row: continue

        cells = row.find_all('td')
        
        if len(cells) < 3: continue 
        
        # Name and Profile ID are in the second TD (index 1)
        player_link = cells[1].find('a', href=True)
        if not player_link: continue
        
        name = player_link.get_text(strip=True)
        js_call = player_link['href']
        id_match = re.search(r'LinkPlayer\((\d+)\)', js_call)
        player_id = id_match.group(1) if id_match else '0'
        player_id_padded = player_id.zfill(6)

        # Rating is in the third TD (index 2)
        rating_text = cells[2].get_text(strip=True).replace(',', '') if len(cells) > 2 else 'N/A'
        
        # Reconstruct Profile URL
        full_profile_url = f"{BASE_URL}{player_url_template_prefix}{player_id_padded}{player_url_template_suffix}"

        player_data.append({
            'name': format_player_name(name), 
            'rating': rating_text, 
            'rank': i,
            'profile_url': full_profile_url
        })

    # 4. Get Next Month URL (Derived by incrementing the current URL)
    next_url = get_next_month_url(current_url)
    
    if not player_data:
        print(f"Warning: Could not extract any player data for {yyyymm_date}. Stopping.")
        return None
        
    return player_data, yyyymm_date, next_url


# --- Main Execution ---

def main():
    # 1. Parse Command-line Inputs and Determine Start Point
    
    start_source = DEFAULT_START_FILE
    max_months = DEFAULT_MAX_MONTHS
    start_url = "" 
    
    os.makedirs(os.path.dirname(TEMP_OUTPUT_FILENAME), exist_ok=True)
    
    if len(sys.argv) == 3:
        # Command-line restart
        csv_filename_input = sys.argv[1]
        max_months_input = sys.argv[2]
        
        try:
            max_months = int(max_months_input)
            if max_months <= 0:
                raise ValueError("MAX_MONTHS must be a positive integer.")
        except ValueError:
            print(f"CRITICAL ERROR: MAX_MONTHS parameter must be a valid positive integer, received: {max_months_input}")
            return

        try:
            start_url = get_last_url_from_csv(csv_filename_input)
            if not start_url:
                 print(f"CRITICAL ERROR: Failed to find URL in last record of '{csv_filename_input}'.")
                 return
            start_source = start_url 
            print(f"Restarting scrape from URL found in '{csv_filename_input}'.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to read starting data from file '{csv_filename_input}': {e}")
            return
            
    elif len(sys.argv) == 1:
        # Default start from local HTML file
        if not os.path.exists(start_source):
             print(f"CRITICAL ERROR: Default start file not found at path: {start_source}")
             return
        
        # Derive the base URL for 1843-01
        start_url = f"{BASE_URL}SingleMonth.asp?Params=184010SSSSS5S113454184301151000005300000210100"
        print(f"Starting scrape from default local file: {start_source} (URL derived for date incrementing).")
        
    else:
        print("Usage (Restart): python monthly_top5_scraper.py <CSV_FILENAME> <MAX_MONTHS>")
        print("Usage (New Run): python monthly_top5_scraper.py")
        return

    # 2. Define CSV Header Fields
    base_fields = ["YYYY-MM", "Rating List URL"]
    player_fields = ["Name", "Rating", "Rank", "Profile URL"]
    fieldnames = base_fields
    for i in range(1, 6):
        for field in player_fields:
            fieldnames.append(f"#{i} {field}")
            
    # 3. Initialize temporary output file (WRITE header ONCE)
    try:
        with open(TEMP_OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write initial CSV header: {e}")
        return

    current_url = start_url
    month_count = 0
    final_end_date = None
    final_start_date = None

    # 4. Main Looping Mechanism (Monthly Iteration)
    while current_url and month_count < max_months:
        
        # Use the local file only for the first month if starting from default file path.
        source_to_fetch = start_source if month_count == 0 and not start_url.startswith('http') else current_url
        
        soup = fetch_content(source_to_fetch)
        
        if not soup:
            print(f"Stopping loop: Failed to fetch/load content for {source_to_fetch}.")
            break
        
        # Scrape and extract.
        scrape_result = scrape_monthly_list(soup, current_url) 
        
        if not scrape_result:
            print(f"Stopping loop: Failed to parse data for {current_url}.")
            break
        
        player_data, yyyymm_date, next_url = scrape_result
        
        if final_start_date is None:
            final_start_date = yyyymm_date
            
        final_end_date = yyyymm_date
        
        # C. Format and append data to CSV
        csv_row: Dict[str, Any] = {
            "YYYY-MM": yyyymm_date, 
            "Rating List URL": current_url
        }
        
        for i in range(5):
            if i < len(player_data):
                player = player_data[i]
                csv_row[f"#{i+1} Name"] = player['name']
                csv_row[f"#{i+1} Rating"] = player['rating']
                csv_row[f"#{i+1} Rank"] = player['rank']
                csv_row[f"#{i+1} Profile URL"] = player['profile_url']
            else:
                # Fill missing ranks with empty strings to maintain column structure
                for field in player_fields:
                    csv_row[f"#{i+1} {field}"] = ""

        try:
            with open(TEMP_OUTPUT_FILENAME, 'a', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(csv_row)
            print(f"SUCCESS: Appended record for {yyyymm_date} (Month {month_count + 1}).")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to append data to CSV: {e}")
            break

        # D. Prepare for next iteration
        current_url = next_url 
        month_count += 1
        
    # 5. Final step: Rename temporary file to final, date-stamped filename
    
    # Filename format: chessmetrics-monthly_top5_YYYY-MM_YYYY-MM.csv
    if final_start_date and final_end_date:
        final_output_filename = os.path.join(
            'data', f"chessmetrics-monthly_top5_{final_start_date}_{final_end_date}.csv"
        )
    else:
        final_output_filename = os.path.join(
            'data', f"chessmetrics-monthly_top5_incomplete_run.csv"
        )

    if os.path.exists(TEMP_OUTPUT_FILENAME):
        if month_count > 0:
            os.rename(TEMP_OUTPUT_FILENAME, final_output_filename)
            print(f"\nCompleted scraping {month_count} monthly lists.")
            print(f"Output saved to: {final_output_filename}")
        else:
            print("No new months scraped. Deleting temporary file.")
            os.remove(TEMP_OUTPUT_FILENAME)

if __name__ == "__main__":
    main()