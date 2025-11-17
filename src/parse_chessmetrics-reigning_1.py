import requests
from bs4 import BeautifulSoup
import csv
import re
import os

# --- Global Constants ---
BASE_URL = "http://chessmetrics.com/cm/CM2/"
OUTPUT_FILENAME = os.path.join('data', 'test2.csv') 
LOCAL_HTML_PATH = os.path.join('data', 'chessmetrics _ Monthly_1843-01.htm')

# --- Utility Functions ---

def format_player_name(full_name: str) -> str:
    """
    Converts 'Firstname(s) Lastname(s)' to 'Lastname(s), Firstname(s)' 
    and removes all periods from the resulting string.
    """
    # 1. Split and reorder name to "Last name, First name"
    parts = full_name.strip().split()
    if len(parts) > 1:
        last_name = parts[-1]
        first_names = ' '.join(parts[:-1])
        result = f"{last_name}, {first_names}"
    else:
        result = full_name # Handle single-word names

    # 2. Remove all periods 
    result = result.replace('.', '').strip()
    
    return result

def read_local_html(file_path: str, from_encoding='windows-1252'):
    """Reads a local HTML file and returns a BeautifulSoup object."""
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding=from_encoding) as f:
            content = f.read()
        return BeautifulSoup(content, 'html.parser')
    except Exception:
        return None

def fetch_page(url: str, from_encoding='windows-1252'):
    """Fetches a URL and returns a BeautifulSoup object."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser', from_encoding=from_encoding)
    except requests.exceptions.RequestException:
        return None

# --- Core Scraping Functions ---

def extract_initial_player_profile(file_path: str) -> tuple[str, str] or None:
    """Reads local HTML, identifies the #1 player, and extracts their profile URL."""
    soup = read_local_html(file_path, from_encoding='windows-1252')
    if not soup: return None
    
    # Find the rank 1 element, which is nested inside <font> and contains "#1"
    rank_one_font = soup.find('font', string=lambda t: t and '#1' in t.strip())
    
    if not rank_one_font:
        return None

    rank_one_row = rank_one_font.find_parent('tr')
    player_link_tag = rank_one_row.find('a', href=True)

    if not player_link_tag:
        return None

    player_name = player_link_tag.get_text(strip=True)
    relative_profile_url = player_link_tag['href']
    
    # Extract Player ID from the javascript:LinkPlayer(ID); call
    js_match = re.search(r"javascript:LinkPlayer\((\d+)\)", relative_profile_url)
    if not js_match:
        return None

    player_id = js_match.group(1)
    
    # Reconstruct the PlayerProfile URL
    # Static parts derived from analyzing the LinkPlayer function for this specific initial page.
    STATIC_PREFIX = "PlayerProfile.asp?Params=184010SSSSS5S"
    STATIC_SUFFIX = "184301151000005300000210100"
    player_id_padded = player_id.zfill(6)
    
    reconstructed_profile_url = f"{STATIC_PREFIX}{player_id_padded}{STATIC_SUFFIX}"
    full_profile_url = BASE_URL + reconstructed_profile_url
    
    return player_name, full_profile_url


def scrape_player_reign(player_name: str, profile_url: str) -> list[dict]:
    """Scrapes historical rank data from the player profile page."""
    soup = fetch_page(profile_url)
    if not soup: return []

    reign_data = []
    
    # 1. Locate the Monthly Ranking History Table
    history_table = None
    rank_text_found = soup.find(string=re.compile(r'.*ranked #\d+ in world.*', re.DOTALL))
    
    if rank_text_found:
        history_table = rank_text_found.find_parent('table')
    
    if not history_table:
        return []

    # 2. Extract Static URL Parameter Parts from the LinkMonth JavaScript function
    static_url_prefix = ""
    static_url_suffix = ""
    
    # This block searches the HTML for the LinkMonth JS code to extract the static parts
    js_code_tag = soup.find('script', string=lambda t: t and 'function LinkMonth' in t)
    if js_code_tag and js_code_tag.string:
        # Regex to find: var NewURL = "PREFIX" + RatingPeriod + "SUFFIX";
        url_match = re.search(
            r'var NewURL = "([^"]*)"\s*\+\s*RatingPeriod\s*\+\s*"([^"]*)";', 
            js_code_tag.string, 
            re.DOTALL
        )
        if url_match:
            static_url_prefix = url_match.group(1)
            static_url_suffix = url_match.group(2)
    
    if not static_url_prefix or not static_url_suffix:
        # Fallback to hardcoded parts derived from the user's JS example if search fails
        static_url_prefix = "SingleMonth.asp?Params=184030SSSSS5S023538" 
        static_url_suffix = "151000005300000210100" 

    # 3. Iterate through rows and collect data
    history_rows = history_table.find_all('tr')
    found_reign_end = False
    
    for row in history_rows:
        cells = row.find_all('td')
        if len(cells) != 2: continue
            
        # Cell 1: Date and LinkMonth JS call
        date_link = cells[0].find('a', href=True)
        if not date_link: continue
        
        js_call = date_link['href']
        js_match = re.search(r'LinkMonth\((\d{4})\.(\d{2})\)', js_call)
        
        if not js_match: continue
        
        year = js_match.group(1)
        month = js_match.group(2)
        date_text = f"{year}-{month}"
        
        # Calculate RatingPeriod: YYYYMM
        rating_period = f"{year}{month}"
        
        # *** URL RECONSTRUCTION FIX ***
        relative_url = f"{static_url_prefix}{rating_period}{static_url_suffix}"
        rating_list_url = BASE_URL + relative_url
        
        # Cell 2: Rating and Rank data
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
            break

    return reign_data

# --- Main Execution ---

def main():
    # 0) Analyze starting page
    initial_data = extract_initial_player_profile(LOCAL_HTML_PATH)
    
    if not initial_data: return 

    player_name, profile_url = initial_data
    
    # 1) & 2) Retrieve player profile and scrape reign data
    player_reign_history = scrape_player_reign(player_name, profile_url)
    
    if not player_reign_history:
        return

    # 3) Format and save output CSV
    fieldnames = ["Lastname(s), Firstname(s)", "YYYY-MM", "Rating List URL", "Rating", "Rank"]
    csv_rows = []
    for data in player_reign_history:
        formatted_name = format_player_name(data['name'])
        csv_rows.append({
            fieldnames[0]: formatted_name, 
            fieldnames[1]: data['date'], 
            fieldnames[2]: data['url'],
            fieldnames[3]: data['rating'], 
            fieldnames[4]: data['rank']
        })

    # Write the data directly to a file using UTF-8 encoding
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
        
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    except Exception:
        pass

if __name__ == "__main__":
    main()