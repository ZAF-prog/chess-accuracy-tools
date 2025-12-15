#!/usr/bin/env python
import sys
import traceback
import argparse
import os
from collections import defaultdict

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter a PGN chess database by eliminating players with fewer than MinGames, iteratively, and standardizing player names."
    )
    parser.add_argument(
        "input_pgn_file",
        help="Path to the input PGN file."
    )
    parser.add_argument(
        "--min_games",
        type=int,
        default=20,
        help="Minimum number of games a player must have to be retained (default: 20)."
    )
    return parser.parse_args()

def create_standardization_map(player_names):
    """
    Creates a map to standardize shorter player names to their longer versions
    (e.g., "Mestrovic,Z" -> "Mestrovic, Zvonimir").

    Args:
        player_names (set): A set of all unique player names found in the PGN.

    Returns:
        dict: A mapping from the shorter name to the standardized longer name.
    """
    # Group names by the part before the comma (Last Name or Last Name, Initials)
    name_groups = defaultdict(list)
    for name in player_names:
        # Normalize: remove spaces around the comma for consistent grouping
        key = name.split(',')[0].strip()
        name_groups[key].append(name)
    
    standardization_map = {}

    for base_name, variations in name_groups.items():
        if len(variations) <= 1:
            continue

        # Find the longest name string to use as the standard
        standard_name = max(variations, key=len)
        
        # Parse the standard name's first name part
        standard_parts = standard_name.split(',')
        standard_first = standard_parts[1].strip() if len(standard_parts) > 1 else ''
        
        for name in variations:
            if name == standard_name:
                continue
                
            # Parse the shorter name's first name part
            name_parts = name.split(',')
            name_first = name_parts[1].strip() if len(name_parts) > 1 else ''
            
            # Only merge if the shorter first name is a prefix of the longer first name
            # This ensures "Smith, A" matches "Smith, Andrew" but not "Smith, Alice"
            if name_first and standard_first.lower().startswith(name_first.lower()):
                standardization_map[name] = standard_name

    return standardization_map

def first_pass_index_games(input_filepath):
    """
    Performs the first pass to index all games by player and byte offset,
    then standardizes the player names.
    """
    game_metadata = []
    current_game_info = {}
    
    # Set to collect all unique player names before standardization
    raw_player_names = set()
    
    # State tracking
    has_seen_moves = False
    
    # Track the position of [Source] tag if it precedes [Event]
    pending_source_offset = None

    print("Indexing games and extracting player information...")

    # Open the file in binary mode for precise byte tracking
    with open(input_filepath, 'rb') as f:
        while True:
            # Track position BEFORE reading the line
            line_start_offset = f.tell()
            line = f.readline()
            
            if not line:
                break
                
            try:
                decoded_line = line.decode('utf-8', errors='ignore').strip()
            except UnicodeDecodeError:
                continue

            # Check for PGN tag pairs
            if decoded_line.startswith('['):
                # Track [Source] tags as potential game start ONLY if we've finished the previous game (seen moves)
                # This prevents [Source] tags inside the *current* game's header from splitting the game.
                if decoded_line.startswith('[Source ') and has_seen_moves:
                    if pending_source_offset is None:
                         pending_source_offset = line_start_offset
                    
                elif decoded_line.startswith('[Event '):
                    # Start of a new game. Finalize the previous one.
                    # Use pending_source_offset if we saw a [Source] after the LAST game's moves
                    game_start = pending_source_offset if pending_source_offset is not None else line_start_offset
                    
                    if game_metadata:
                        prev_index = len(game_metadata) - 1
                        game_metadata[prev_index]['end_offset'] = game_start
                        
                    # Initialize new game info - start at Source tag or Event tag position
                    current_game_info = {
                        'white': None, 
                        'black': None, 
                        'start_offset': game_start,
                        'keep': True,
                        'end_offset': None
                    }
                    game_metadata.append(current_game_info)
                    
                    # Reset state for new game
                    pending_source_offset = None
                    has_seen_moves = False
                
                # Extract White and Black player names
                elif decoded_line.startswith('[White '):
                    parts = decoded_line.split('"')
                    if len(parts) >= 2:
                        player = parts[1]
                    else:
                        # Fallback for malformed tags without quotes
                        player = decoded_line.replace('[White', '').replace(']', '').strip()
                        
                    current_game_info['white'] = player
                    raw_player_names.add(player)
                    
                elif decoded_line.startswith('[Black '):
                    parts = decoded_line.split('"')
                    if len(parts) >= 2:
                        player = parts[1]
                    else:
                        # Fallback for malformed tags without quotes
                        player = decoded_line.replace('[Black', '').replace(']', '').strip()

                    current_game_info['black'] = player
                    raw_player_names.add(player)
            
            # If line is not empty and not a tag, it's likely a move line
            elif decoded_line:
                has_seen_moves = True
                # If we see moves, any pending source offset from BEFORE these moves is invalid/stale
                # (though logic should prevent setting it unless has_seen_moves was True from PREVIOUS game)
                if pending_source_offset is not None:
                     # This implies we saw [Source] -> Moves -> [Event].
                     # Standard PGN implies [Source] was part of the *upcoming* game?
                     # No, if we see moves, then the [Source] we saw earlier was just a header tag for THIS game (or previous).
                     # PGN structure: [Tags] \n\n Moves \n\n [Tags]
                     # If we are in Moves, we are clearly NOT in the "Pre-Event Source Tag" zone of the NEXT game.
                     pending_source_offset = None

        # Finalize the last game's end offset (EOF)
        if game_metadata and game_metadata[-1]['end_offset'] is None:
             game_metadata[-1]['end_offset'] = f.tell()
    
    print(f"Total games indexed: {len(game_metadata)}")
    print(f"Unique player names found (before standardization): {len(raw_player_names)}")

    # --- Standardization Step ---
    print("Standardizing player names (merging shorter versions into longer versions)...")
    standardization_map = create_standardization_map(raw_player_names)
    
    if standardization_map:
        # Apply the map to the game metadata
        for game in game_metadata:
            game['white'] = standardization_map.get(game['white'], game['white'])
            game['black'] = standardization_map.get(game['black'], game['black'])
        
        # Re-check unique player count after standardization
        final_player_names = set(game['white'] for game in game_metadata)
        final_player_names.update(game['black'] for game in game_metadata)
        
        print(f"Unique player names after standardization: {len(final_player_names)}")
        print(f"Total standardizations performed: {len(standardization_map)}")
    else:
        print("No name variations requiring standardization were found.")

    return game_metadata

def iterative_filter_games(game_metadata, min_games):
    """
    Iteratively removes games played by players who do not meet the 
    min_games threshold, until all remaining players are compliant.
    """
    iteration = 0
    eliminated_players = set()
    
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # 1. Recalculate game counts based on 'keep=True'
        player_game_counts = defaultdict(int)
        for game in game_metadata:
            if game['keep']:
                player_game_counts[game['white']] += 1
                player_game_counts[game['black']] += 1

        # 2. Identify players to eliminate
        newly_eliminated = set()
        for player, count in player_game_counts.items():
            # A player is eliminated if their game count is below the minimum AND
            # they are not already in the set of players previously eliminated
            if count < min_games and player not in eliminated_players:
                newly_eliminated.add(player)

        if not newly_eliminated:
            # Filtering complete
            print(f"No further players eliminated. Filtering complete.")
            break
        
        # 3. Update game metadata: mark games as 'keep=False'
        games_removed_count = 0
        for game in game_metadata:
            if game['keep']:
                if game['white'] in newly_eliminated or game['black'] in newly_eliminated:
                    game['keep'] = False
                    games_removed_count += 1
        
        eliminated_players.update(newly_eliminated)
        
        # Log results for the iteration
        current_active_players = sum(1 for count in player_game_counts.values() if count >= min_games)
        
        print(f"Players newly eliminated: {len(newly_eliminated)}")
        print(f"Total eliminated players: {len(eliminated_players)}")
        print(f"Games marked for removal: {games_removed_count}")
        print(f"Remaining players with >= {min_games} games: {current_active_players}")
        
    return game_metadata

def write_output_pgn(input_filepath, output_filepath, game_metadata):
    """
    Performs the second pass to read kept games from the input file
     and write them to the output file.
    """
    kept_games_count = sum(1 for game in game_metadata if game['keep'])
    
    print(f"\nWriting {kept_games_count} games to output file: {output_filepath}")

    with open(input_filepath, 'rb') as infile, open(output_filepath, 'wb') as outfile:
        # Create a temporary string buffer to hold the modified metadata lines
        temp_buffer = b''
        
        for game in game_metadata:
            if game['keep']:
                start_offset = game['start_offset']
                end_offset = game['end_offset']
                
                if end_offset is None or end_offset <= start_offset:
                    print(f"Warning: Skipping game at offset {start_offset} due to invalid length.")
                    continue

                game_length = end_offset - start_offset
                
                # Seek and read the raw game content (tags + moves)
                infile.seek(start_offset)
                raw_game_content = infile.read(game_length).decode('utf-8', errors='ignore')
                
                # Rebuild the game tags with standardized names
                segments = []
                current_tags = []
                current_moves = []
                state = "TAGS" # TAGS or MOVES

                for line in raw_game_content.splitlines():
                    striped = line.strip()
                    if striped.startswith('['):
                        if state == "MOVES":
                            # Transition from MOVES -> TAGS implies a new game block started in this chunk
                            segments.append( (current_tags, current_moves) )
                            current_tags = []
                            current_moves = []
                            state = "TAGS"
                        
                        # Handle standardization for the PRIMARY game (first segment)
                        # or just pass through for subsequent embedded games
                        if not segments and line.startswith('[White '):
                            line = f'[White "{game["white"]}"]'
                        elif not segments and line.startswith('[Black '):
                            line = f'[Black "{game["black"]}"]'
                        
                        current_tags.append(line)
                    elif striped:
                        # Non-empty, non-tag line -> Move
                        state = "MOVES"
                        current_moves.append(line)
                    # Ignore empty lines, we will generate structural newlines
                
                # Append final segment
                segments.append( (current_tags, current_moves) )

                # Write all segments
                for tags, moves in segments:
                    if not tags and not moves: continue
                    rebuilt_pgn = '\n'.join(tags) + '\n\n' + '\n'.join(moves) + '\n\n'
                    outfile.write(rebuilt_pgn.encode('utf-8'))

    print("Output complete.")
    return kept_games_count


def main():
    """Main function to run the filtering process."""
    args = parse_args()
    input_file = args.input_pgn_file
    min_games = args.min_games

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'", file=sys.stderr)
        sys.exit(1)

    # 1. Prepare output filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_MinGames={min_games}{ext}"
    
    try:
        # 2. First Pass: Indexing and Standardization
        game_metadata = first_pass_index_games(input_file)
        
        if not game_metadata:
            print("No games found in the input file. Exiting.")
            sys.exit(0)

        # 3. Iterative Filtering
        final_metadata = iterative_filter_games(game_metadata, min_games)
        
        # 4. Second Pass: Output (writing standardized tags)
        kept_games_count = write_output_pgn(input_file, output_file, final_metadata)
        
        print(f"\nSuccessfully created file: '{output_file}'")
        print(f"Original games: {len(game_metadata)}, Retained games: {kept_games_count}")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
