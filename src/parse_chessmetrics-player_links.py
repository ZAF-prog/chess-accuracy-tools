import requests
from bs4 import BeautifulSoup
import re

def reconstruct_player_links(url: str):
    """
    Fetches the FindPlayerByPrefix page, analyzes the embedded 'LinkPlayer' 
    JavaScript function, and reconstructs the full PlayerProfile.asp URL for 
    every player link on the page.
    """
    # Base URL for constructing the absolute link
    BASE_URL = "http://chessmetrics.com/cm/CM2/"
    
    try:
        # 1. Fetch the HTML content
        print(f"[DEBUG] Attempting to fetch URL: {url}")
        response = requests.get(url)
        response.raise_for_status() 
        print("[DEBUG] Page fetched successfully (Status 200).")

        # 2. Parse the HTML using 'windows-1252' (the fix for accented characters)
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='windows-1252')
        
        # 3. Find the JavaScript code block containing the LinkPlayer function
        js_code = ""
        for script in soup.find_all('script'):
            if script.string and 'function LinkPlayer' in script.string:
                js_code = script.string
                break

        if not js_code:
            return "Error: Could not find the JavaScript function 'LinkPlayer' in the page."
        print("[DEBUG] LinkPlayer JavaScript code block found.")


        # 4. Analyze the LinkPlayer function body to extract static URL components
        
        # A. Find the function argument name (e.g., 'NewPlayerCode')
        arg_match = re.search(r"function LinkPlayer\((\w+)\)", js_code)
        if not arg_match:
            return "Error: Could not determine the parameter name for LinkPlayer."
        player_code_var = arg_match.group(1)

        # B. Find the concatenation in the JS code: var NewURL = "STATIC_PART_1" + NewPlayerCode + "STATIC_PART_2";
        var_escaped = re.escape(player_code_var)
        url_match = re.search(
            r'var NewURL = "([^"]*)"\s*\+\s*' + var_escaped + r'\s*\+\s*"([^"]*)";', 
            js_code, 
            re.DOTALL
        )

        if not url_match:
            print("[DEBUG] Failed to match URL concatenation regex.")
            # Print the relevant JS section for manual inspection if needed
            js_snippet = js_code.split('function LinkPlayer')[1].split('}')[0]
            print(f"[DEBUG] JS Snippet analyzed:\n{js_snippet}")
            return "Error: Could not extract the static URL parts from LinkPlayer function body."

        static_part_1 = url_match.group(1)
        static_part_2 = url_match.group(2)
        
        print(f"[DEBUG] JS Analysis: Static URL Part 1: '{static_part_1}'")
        print(f"[DEBUG] JS Analysis: Static URL Part 2: '{static_part_2}'")

        # 5. Find all player links calling LinkPlayer
        player_links = []
        
        js_link_regex = re.compile(r"javascript:LinkPlayer\('(\d+)'\);")
        all_links = soup.find_all('a', href=js_link_regex)
        
        if not all_links:
            return "Error: Found JS function, but no links calling LinkPlayer()."
            
        print(f"[DEBUG] Found {len(all_links)} links calling LinkPlayer(). Starting reconstruction...")

        # 6. Reconstruct the full profile URL for each player
        for link in all_links:
            href = link['href']
            player_name = link.get_text(strip=True)
            
            match = js_link_regex.match(href)
            
            if match:
                player_id = match.group(1)
                
                # Reconstruct the full link: Part1 + ID + Part2
                relative_url = f"{static_part_1}{player_id}{static_part_2}"
                full_link = BASE_URL + relative_url
                
                player_links.append({
                    'name': player_name,
                    'id': player_id,
                    'link': full_link
                })

        # 7. Output Results (limited to first 10 for readability)
        if player_links:
            print(f"\nSuccessfully reconstructed {len(player_links)} player profile links:")
            print("----------------------------------------------------------------------")
            for item in player_links[:10]:
                print(f"Name: {item['name']:<25} | ID: {item['id']:<5} | Link: {item['link']}")
            if len(player_links) > 10:
                print(f"... and {len(player_links) - 10} more links.")
            print("----------------------------------------------------------------------")
        
        return player_links

    except requests.exceptions.RequestException as e:
        return f"Network Error: Failed to fetch URL: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# URL to analyze (search results for prefix group 363)
url_to_analyze = "http://chessmetrics.com/cm/CM2/FindPlayerByPrefix.asp?Params=197701SSSSS5S112053197712151000000000036310100"

# --- FUNCTION CALL UNCOMMENTED ---
reconstruct_player_links(url_to_analyze)