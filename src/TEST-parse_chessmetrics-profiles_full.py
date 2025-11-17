import requests
from bs4 import BeautifulSoup
import re
import csv
import time
import random
import os

# --- Global Constants ---
BASE_URL = "http://chessmetrics.com/cm/CM2/"
OUTPUT_FILENAME = os.path.join('data', 'chessmetrics_player_profiles.csv') 
LOCAL_HTML_PATH = os.path.join('data', 'chessmetrics _ Find A Player.htm')

# --- Utility Functions ---

def throttle():
    delay = random.uniform(1.6, 2.4)
    time.sleep(delay)

def format_player_name(full_name: str) -> str:
    """Keeps the original name order and removes all periods."""
    result = full_name.replace('.', '').strip()
    return result

def read_local_html(file_path: str, from_encoding='windows-1252'):
    """Reads a local HTML file and returns a BeautifulSoup object."""
    try:
        if not os.path.exists(file_path):
            print(f"[DEBUG] CRITICAL: Local file not found at path: {os.path.abspath(file_path)}")
            return None
        with open(file_path, 'r', encoding=from_encoding) as f:
            content = f.read()
        print(f"[DEBUG] Successfully read local file: {file_path}")
        return BeautifulSoup(content, 'html.parser')
    except Exception as e:
        print(f"[DEBUG] CRITICAL: Error reading local file: {e}")
        return None

def fetch_page(url: str, from_encoding='windows-1252'):
    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser', from_encoding=from_encoding)
    except requests.exceptions.RequestException as e:
        # Note the specific URL that failed to fetch
        print(f"[DEBUG] Network Error: Failed to fetch {url[:60]}... Reason: {e}")
        return None

# --- Core Scraping Functions (Unchanged logic) ---

def extract_prefix_links(file_path: str) -> list:
    """Reconstructs 'prefixgroup' links from the local file."""
    soup = read_local_html(file_path, from_encoding='windows-1252')
    if not soup: return []
    prefix_links = []
    js_code = soup.find('script', string=lambda t: t and 'function FindPrefixGroup' in t)
    if not js_code or not js_code.string: return []
    url_match = re.search(r'var NewURL = "([^"]*)"\s*\+\s*NewPrefixGroup\s*\+\s*"([^"]*)";', js_code.string, re.DOTALL)
    if not url_match: return []
    STATIC_URL_PREFIX = url_match.group(1)
    STATIC_URL_SUFFIX = url_match.group(2)
    js_regex = re.compile(r"javascript:FindPrefixGroup\('(\d+)'\);")
    all_links = soup.find_all('a', href=js_regex)
    for link in all_links:
        href = link['href']
        match = js_regex.match(href)
        if match:
            prefix_group_id = match.group(1)
            relative_url = f"{STATIC_URL_PREFIX}{prefix_group_id}{STATIC_URL_SUFFIX}"
            full_link = BASE_URL + relative_url
            prefix_links.append(full_link)
    return prefix_links


def reconstruct_player_links(prefix_url: str) -> list:
    """Reconstructs the full PlayerProfile.asp URL for every player on a page."""
    soup = fetch_page(prefix_url)
    if not soup: return []
    player_links = []
    js_code_tag = soup.find('script', string=lambda t: t and 'function LinkPlayer' in t)
    if not js_code_tag or not js_code_tag.string: return []
    js_code = js_code_tag.string
    arg_match = re.search(r"function LinkPlayer\((\w+)\)", js_code)
    if not arg_match: return []
    player_code_var = arg_match.group(1)
    var_escaped = re.escape(player_code_var)
    url_match = re.search(r'var NewURL = "([^"]*)"\s*\+\s*' + var_escaped + r'\s*\+\s*"([^"]*)";', js_code, re.DOTALL)
    if not url_match: return []
    static_part_1 = url_match.group(1)
    static_part_2 = url_match.group(2)
    js_link_regex = re.compile(r"javascript:LinkPlayer\('(\d+)'\);")
    all_links = soup.find_all('a', href=js_link_regex)
    for link in all_links:
        href = link['href']
        player_name = link.get_text(strip=True)
        match = js_link_regex.match(href)
        if match:
            player_id = match.group(1)
            relative_url = f"{static_part_1}{player_id}{static_part_2}"
            full_link = BASE_URL + relative_url
            player_links.append({'name': player_name, 'link': full_link})
    return player_links

# --- Main Execution ---

def main():
    all_player_data = []

    # 1) Reconstruct all 'prefixgroup' links from local file
    prefix_links = extract_prefix_links(LOCAL_HTML_PATH)
    
    if not prefix_links:
        print("[DEBUG] CRITICAL: Prefix links list is empty. Script exiting.")
        return

    print(f"[DEBUG] Found {len(prefix_links)} prefix pages to scrape.")

    # 2) Traverse ALL prefix pages with throttling (FULL RUN)
    for i, prefix_url in enumerate(prefix_links):
        if i > 0:
            throttle()
        
        # 3) Reconstruct PlayerProfile.asp URLs from the fetched page
        players_on_page = reconstruct_player_links(prefix_url)
        
        # DEBUG: Report number of players found on the current page
        if players_on_page:
            print(f"[DEBUG] Scraped {len(players_on_page)} players from page {i + 1}/{len(prefix_links)}")
        
        all_player_data.extend(players_on_page)

    print(f"[DEBUG] Finished scraping. Total players collected: {len(all_player_data)}")

    # 4) Construct and write output CSV
    fieldnames = ["Lastname(s), Firstname(s)", "PlayerProfile.asp URL"]
    csv_rows = []
    for player in all_player_data:
        formatted_name = format_player_name(player['name'])
        csv_rows.append({fieldnames[0]: formatted_name, fieldnames[1]: player['link']})

    try:
        os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
        print(f"[DEBUG] Attempting to write {len(csv_rows)} rows to '{OUTPUT_FILENAME}'...")
        
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
            
        print(f"[DEBUG] SUCCESS: File written to '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"[DEBUG] CRITICAL ERROR: Failed to write file: {e}")

if __name__ == "__main__":
    main()