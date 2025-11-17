import requests
from bs4 import BeautifulSoup
import csv
import re
import os
import io
import time
import random
import sys

# --- Global Constants ---
BASE_URL = "http://chessmetrics.com/cm/CM2/"
# Temporary file is created inside a 'data' folder relative to the script's execution path
TEMP_OUTPUT_FILENAME = os.path.join('data', 'chessmetrics-reigning_temp.csv') 

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
        last_name = parts[-1]
        first_names = ' '.join(parts[:-1])
        result = f"{last_name}, {first_names}"
    else:
        result = full_name
    
    return result.replace('.', '').strip()

def fetch_page(url: str, from_encoding='windows-1252'):
    """Fetches a URL and returns a BeautifulSoup object, applying throttle."""
    throttle() 
    try:
        # Chessmetrics uses Windows-1252 encoding
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser', from_encoding=from_encoding)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_input_csv_line(csv_line_string: str) -> tuple[str, str]:
    """
    Parses a single CSV line string to extract the starting URL and Date1.
    This function expects a full, quoted CSV line of data from the input file.
    """
    # Use io.StringIO to treat the string as a file for the CSV reader
    reader = csv.reader(io.StringIO(csv_line_string))
    try:
        row = next(reader)
    except StopIteration:
        raise ValueError("Input CSV line is empty.")
    
    # Expected columns are typically: Name, Date, URL, Rating, Rank. We need Date (index 1) and URL (index 2).
    if len(row) < 3:
        raise ValueError(f"Input CSV line has too few columns ({len(row)}). Expected at least 3: Name, Date, URL.")

    date1 = row[1].strip()
    url = row[2].strip()
    
    # Check for YYYY-MM format
    if not re.match(r"\d{4}-\d{2}", date1):
        raise ValueError(f"Invalid date format in input line: {date1}. Expected YYYY-MM.")

    return url, date1

def get_last_record_from_file(filepath: str) -> tuple[str, str]:
    """
    Reads the specified CSV file and extracts the URL and Date from the last record.
    Returns (url, date_start)
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
        raise ValueError(f"Input file '{filepath}' contains no data records.")

    # Parse the CSV line content
    return parse_input_csv_line(last_data_line)


# --- Core Scraping Functions ---

def find_player_info_from_monthly_list_soup(soup: BeautifulSoup) -> tuple[str, str, str] | None:
    """
    Extracts the #1 player's name and profile URL from a monthly list soup object.
    This is used to determine the next player whose reign needs to be scraped.
    """
    
    rank_one_font = soup.find('font', string=lambda t: t and '#1' in t.strip())
    if not rank_one_font: return None

    rank_one_row = rank_one_font.find_parent('tr')
    player_link_tag = rank_one_row.find('a', href=True)
    if not player_link_tag: return None

    player_name = player_link_tag.get_text(strip=True)
    relative_profile_url = player_link_tag['href']
    js_match = re.search(r"javascript:LinkPlayer\((\d+)\)", relative_profile_url)
    if not js_match: return None

    player_id = js_match.group(1)
    player_id_padded = player_id.zfill(6)
    
    # Extract the static parts of the PlayerProfile URL from the LinkPlayer JS function
    js_code_tag = soup.find('script', string=lambda t: t and 'function LinkPlayer' in t)
    if not js_code_tag or not js_code_tag.string: return None
    
    url_match = re.search(
        r'var NewURL = "([^"]*)"\s*\+\s*NewPlayerID\s*\+\s*"([^"]*)";', 
        js_code_tag.string, 
        re.DOTALL
    )
    if not url_match: return None
    
    static_prefix = url_match.group(1)
    static_suffix = url_match.group(2)
    
    reconstructed_profile_url = f"{static_prefix}{player_id_padded}{static_suffix}"
    full_profile_url = BASE_URL + reconstructed_profile_url
    
    return player_name, full_profile_url, None 

def scrape_player_reign(player_name: str, profile_url: str) -> tuple[list[dict], str] | tuple[None, None]:
    """
    Scrapes historical rank data from the player profile page, collects consecutive #1 records,
    and returns the URL of the first month the player dropped from #1.
    """
    print(f"Scraping profile for {player_name}...")
    soup = fetch_page(profile_url)
    if not soup: return None, None

    reign_data = []
    
    history_table = None
    # Look for the table containing the historical rank data
    rank_text_found = soup.find(string=re.compile(r'.*ranked #\d+ in world.*', re.DOTALL))
    if rank_text_found:
        history_table = rank_text_found.find_parent('table')
    if not history_table:
        return None, None
        
    # Extract LinkMonth static parts from the JS function on the profile page
    static_url_prefix = ""
    static_url_suffix = ""
    js_code_tag = soup.find('script', string=lambda t: t and 'function LinkMonth' in t)
    if js_code_tag and js_code_tag.string:
        url_match = re.search(
            r'var NewURL = "([^"]*)"\s*\+\s*RatingPeriod\s*\+\s*"([^"]*)";', 
            js_code_tag.string, 
            re.DOTALL
        )
        if url_match:
            static_url_prefix = url_match.group(1)
            static_url_suffix = url_match.group(2)
    if not static_url_prefix or not static_url_suffix:
        return None, None

    history_rows = history_table.find_all('tr')
    found_reign_end = False
    next_start_url = None
    
    for row in history_rows:
        cells = row.find_all('td')
        if len(cells) != 2: continue
            
        date_link = cells[0].find('a', href=True)
        if not date_link: continue
        
        js_call = date_link['href']
        js_match = re.search(r'LinkMonth\((\d{4})\.(\d{2})\)', js_call)
        if not js_match: continue
        
        year = js_match.group(1)
        month = js_match.group(2)
        date_text = f"{year}-{month}"
        rating_period = f"{year}{month}"
        
        # Correctly use static_url_suffix
        relative_url = f"{static_url_prefix}{rating_period}{static_url_suffix}" 
        rating_list_url = BASE_URL + relative_url
        
        data_text = cells[1].get_text(strip=True)
        
        rating_match = re.search(r'Rating: (\d+)', data_text)
        rating_text = rating_match.group(1) if rating_match else 'N/A'
        
        rank_match = re.search(r'ranked #(\d+)', data_text)
        rank_text = rank_match.group(1) if rank_match else 'N/A'
        
        if rank_text == 'N/A': continue
        
        current_rank = int(rank_text)

        is_current_rank_one = (current_rank == 1)
        
        if is_current_rank_one and not found_reign_end:
            # Player is #1 and we haven't found the end yet, so this is part of the reign
            reign_data.append({
                'name': player_name, 'date': date_text, 'url': rating_list_url,
                'rating': rating_text, 'rank': current_rank
            })
            
        elif not is_current_rank_one and not found_reign_end:
            # Player dropped from #1. This is the first month of the drop.
            reign_data.append({
                'name': player_name, 'date': date_text, 'url': rating_list_url,
                'rating': rating_text, 'rank': current_rank
            })
            found_reign_end = True
            next_start_url = rating_list_url # This URL is the starting point for the *next* player's reign search
            break

    return reign_data, next_start_url

# --- Main Execution ---

def main():
    # 1. Parse Command-line Inputs (using sys.argv)
    if len(sys.argv) != 3:
        # Corrected usage message: expects filename instead of CSV line string
        print("Usage: python parse_chessmetrics-reigning_restart.py <CSV_FILENAME> <MAX_REIGNS>")
        print("Example: python parse_chessmetrics-reigning_restart.py data/chessmetrics-reigning_1843-01-1851-01.csv 2")
        return

    csv_filename_input = sys.argv[1]
    max_reigns_input = sys.argv[2]
    
    try:
        max_reigns = int(max_reigns_input)
        if max_reigns <= 0:
            raise ValueError("MAX_REIGNS must be a positive integer.")
    except ValueError:
        print(f"CRITICAL ERROR: MAX_REIGNS parameter must be a valid positive integer, received: {max_reigns_input}")
        return

    try:
        # Get the monthly list URL and date from the last record of the provided CSV file.
        # This URL is the first page scraped in the new run.
        initial_monthly_list_url, date_from_old_file = get_last_record_from_file(csv_filename_input)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read starting data from file '{csv_filename_input}': {e}")
        return

    print(f"Restarting scrape from monthly list URL for date {date_from_old_file}. Scraping {max_reigns} subsequent reigns.")
    
    fieldnames = ["Lastname(s), Firstname(s)", "YYYY-MM", "Rating List URL", "Rating", "Rank"]
    
    # 2. Initialize temporary output file (WRITE header ONCE)
    os.makedirs(os.path.dirname(TEMP_OUTPUT_FILENAME), exist_ok=True)
    try:
        with open(TEMP_OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write initial CSV header: {e}")
        return

    current_monthly_list_url = initial_monthly_list_url
    reign_count = 0
    # Initialize date variables
    new_scrape_start_date = None # Tracks the date of the very first record written in this run (YYYY-MM)
    date_end = date_from_old_file # Tracks the date of the last record written in this run (YYYY-MM)

    # 3. Main Looping Mechanism (Online Fetching)
    while current_monthly_list_url and reign_count < max_reigns:
        print(f"\n--- Starting Reign #{reign_count + 1} Cycle (Limit: {max_reigns}) ---")
        
        # A. Find the #1 player on the new monthly list URL
        list_soup = fetch_page(current_monthly_list_url)
        if not list_soup:
            print("Stopping loop: Failed to fetch monthly list.")
            break
            
        initial_data = find_player_info_from_monthly_list_soup(list_soup)
        
        if not initial_data:
            print("Stopping loop: Could not identify next reigning player or profile link.")
            break
        
        player_name, profile_url, _ = initial_data
        
        # B. Scrape the player's reign history and find the next start URL
        reign_history, next_start_url = scrape_player_reign(player_name, profile_url)
        
        if not reign_history:
            print("Stopping loop: Failed to scrape reign history.")
            break

        # C. Format and append data to CSV
        csv_rows = []
        for data in reign_history:
            formatted_name = format_player_name(data['name'])
            csv_rows.append({
                fieldnames[0]: formatted_name, 
                fieldnames[1]: data['date'], 
                fieldnames[2]: data['url'],
                fieldnames[3]: data['rating'], 
                fieldnames[4]: data['rank']
            })
            
            # Set the actual start date of the new scrape only once (from the first record in the first reign_history list)
            if new_scrape_start_date is None:
                new_scrape_start_date = data['date']
                
            date_end = data['date'] # Update date_end with the latest date

        
        try:
            with open(TEMP_OUTPUT_FILENAME, 'a', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(csv_rows)
            print(f"SUCCESS: Appended {len(csv_rows)} rows for {player_name} (Total reigns: {reign_count + 1}).")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to append data to CSV: {e}")
            break

        # D. Prepare for next iteration
        current_monthly_list_url = next_start_url
        reign_count += 1
        
    # 4. Final step: Rename temporary file to final, date-stamped filename
    
    # Use the actual first date scraped, or the date of the input file's last record if no new data was scraped
    if new_scrape_start_date is not None:
        final_start_date = new_scrape_start_date
    else:
        # If no new data was scraped, use the input file's last date as a fallback for the start date.
        final_start_date = date_from_old_file

    # Output filename format: '~/data/chessmetrics-reigning_YYYY-MM-YYYY-MM.csv'
    # The date variables already contain the hyphen, fulfilling the formatting requirement.
    final_output_filename = os.path.join(
        'data', f"chessmetrics-reigning_{final_start_date}_{date_end}.csv"
    )

    if os.path.exists(TEMP_OUTPUT_FILENAME):
        # Only rename if we successfully scraped at least one reign (reign_count > 0)
        if reign_count > 0:
            os.rename(TEMP_OUTPUT_FILENAME, final_output_filename)
        else:
            print("No new reigns scraped. Deleting temporary file.")
            os.remove(TEMP_OUTPUT_FILENAME)


    print(f"\nCompleted scraping {reign_count} additional reigns.")
    if reign_count > 0:
        print(f"Output saved to: {final_output_filename}")

if __name__ == "__main__":
    main()