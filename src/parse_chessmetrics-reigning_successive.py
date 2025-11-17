import requests
from bs4 import BeautifulSoup
import csv
import re
import os
import time
import random

# --- Global Constants ---
BASE_URL = "http://chessmetrics.com/cm/CM2/"
OUTPUT_FILENAME = os.path.join('data', 'chessmetrics-reigning_successive.csv') 
# NEW: Local file path for the initial monthly list page
LOCAL_START_PATH = os.path.join('data', 'chessmetrics _ Monthly_1843-01.htm')

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

def read_local_html(file_path: str, from_encoding='windows-1252'):
    """Reads a local HTML file and returns a BeautifulSoup object."""
    try:
        if not os.path.exists(file_path):
            print(f"CRITICAL: Local file not found at path: {os.path.abspath(file_path)}")
            return None
        with open(file_path, 'r', encoding=from_encoding) as f:
            content = f.read()
        return BeautifulSoup(content, 'html.parser')
    except Exception as e:
        print(f"CRITICAL: Error reading local file: {e}")
        return None

def fetch_page(url: str, from_encoding='windows-1252'):
    """Fetches a URL and returns a BeautifulSoup object, applying throttle."""
    throttle() # <-- Throttle before fetching page
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser', from_encoding=from_encoding)
    except requests.exceptions.RequestException:
        return None

def find_player_info_from_monthly_list_soup(soup: BeautifulSoup) -> tuple[str, str, str] or None:
    """Extracts #1 player name, profile URL, and params from a monthly list soup object."""
    
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
    
    # Extract the full URL parameters from the LinkPlayer JS function on this page
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
    
    profile_url_params = static_prefix.replace('PlayerProfile.asp?Params=', '') + player_id_padded + static_suffix
    reconstructed_profile_url = f"{static_prefix}{player_id_padded}{static_suffix}"
    full_profile_url = BASE_URL + reconstructed_profile_url
    
    return player_name, full_profile_url, profile_url_params


def scrape_player_reign(player_name: str, profile_url: str) -> tuple[list[dict], str] or tuple[None, None]:
    """Scrapes historical rank data from the player profile page."""
    print(f"Scraping profile for {player_name}...")
    soup = fetch_page(profile_url)
    if not soup: return None, None

    reign_data = []
    
    # 1. Locate the Monthly Ranking History Table
    history_table = None
    rank_text_found = soup.find(string=re.compile(r'.*ranked #\d+ in world.*', re.DOTALL))
    if rank_text_found:
        history_table = rank_text_found.find_parent('table')
    if not history_table:
        return None, None
        
    # 2. Extract Static URL Parameter Parts for LinkMonth
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

    # 3. Iterate through rows and collect data
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
            reign_data.append({
                'name': player_name, 'date': date_text, 'url': rating_list_url,
                'rating': rating_text, 'rank': current_rank
            })
            
        elif not is_current_rank_one and not found_reign_end:
            reign_data.append({
                'name': player_name, 'date': date_text, 'url': rating_list_url,
                'rating': rating_text, 'rank': current_rank
            })
            found_reign_end = True
            next_start_url = rating_list_url
            break

    return reign_data, next_start_url

# --- Main Execution ---

def main():
    # SET LIMIT HERE
    #MAX_REIGNS = 3 
    MAX_REIGNS = 99 
    
    fieldnames = ["Lastname(s), Firstname(s)", "YYYY-MM", "Rating List URL", "Rating", "Rank"]
    
    # 1. Initialize output file (WRITE header ONCE)
    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write initial CSV header: {e}")
        return

    current_monthly_list_url = None
    reign_count = 0
    
    # ----------------------------------------------
    # STAGE 1: Process Initial Reign from Local File
    # ----------------------------------------------
    print(f"--- Starting Reign #1 Cycle (Limit: {MAX_REIGNS}) ---")
    print(f"Reading initial monthly list from local file: {LOCAL_START_PATH}")
    
    local_soup = read_local_html(LOCAL_START_PATH)
    if not local_soup:
        print("Stopping script: Failed to read local starting file.")
        return

    initial_data = find_player_info_from_monthly_list_soup(local_soup)
    if not initial_data:
        print("Stopping script: Could not find #1 player in local starting file.")
        return

    player_name, profile_url, _ = initial_data
    
    # Scrape the first player's reign (John Cochrane)
    reign_history, next_start_url = scrape_player_reign(player_name, profile_url)
    
    if not reign_history:
        print("Stopping script: Failed to scrape initial reign history.")
        return
        
    # Write the first reign's data
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
    
    try:
        with open(OUTPUT_FILENAME, 'a', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(csv_rows)
        print(f"SUCCESS: Appended {len(csv_rows)} rows for {player_name} (Total reigns: 1).")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to append data to CSV: {e}")
        return
    
    # Initialize loop variables for Stage 2
    current_monthly_list_url = next_start_url
    reign_count = 1
    
    # ----------------------------------------------
    # STAGE 2: Main Looping Mechanism (Online Fetching)
    # ----------------------------------------------
    while current_monthly_list_url and reign_count < MAX_REIGNS:
        print(f"\n--- Starting Reign #{reign_count + 1} Cycle (Limit: {MAX_REIGNS}) ---")
        
        # A. Find the #1 player on the new monthly list URL (which is online)
        initial_data = find_player_info_from_monthly_list_soup(fetch_page(current_monthly_list_url))
        
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
        
        try:
            with open(OUTPUT_FILENAME, 'a', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerows(csv_rows)
            print(f"SUCCESS: Appended {len(csv_rows)} rows for {player_name} (Total reigns: {reign_count + 1}).")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to append data to CSV: {e}")
            break

        # D. Prepare for next iteration
        current_monthly_list_url = next_start_url
        reign_count += 1
        
    print(f"\nCompleted scraping {reign_count} reigns. Output saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    main()