package dataextract;
/*
 *  This file is part of uci-analyser: a UCI-based Chess Game Analyser
 *  Copyright (C) 2013-2017 David J. Barnes
 *
 *  uci-analyser is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  uci-analyser is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with uci-analyser.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  David J. Barnes may be contacted as d.j.barnes@kent.ac.uk
 *  https://www.cs.kent.ac.uk/people/staff/djb/
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generate statistics for games.
 * 
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class GenStats {
    // Pseudo player name to allow matching of all games.
    private static final String MATCH_ANY_PLAYER = "<WhiteOrBlack>";
    
    // Player names to be matched.
    private final List<String> players;
    // Game IDs to be matched.
    private final List<String> ids;
    // Game Hashcodes to be matched.
    private final List<String> hashCodes;
    private int minLength = 10;
    private boolean showFullScores = false;
    
    /**
     * The logic for matching based on scores is as follows:
     * 
     * If CVThreshold == 0 then
     *     if !thresholdSet then match everything
     *     else only match games whose mean (AE) is above the threshold.
     *     end if
     * else
     *     match games that have thresholdPercentage percent of their
     *     moves at least as good as the lowThreshold, regardless of
     *     whether an explicit threshold has been set or not.
     * end if
     */
    private boolean showAccuracy = false;
    // Cut-off percentage of moves below the threshold.
    private double CVThreshold = 0;    
    // Whether a lowThreshold has been set.
    private boolean thresholdSet = false;
    // Cut-off point for outputting games.
    private double AEThreshold = 0;
    
    private double randomThreshold = 0;
    private final Random rand = new Random();

    public GenStats() {
        players = new ArrayList<>();
        ids = new ArrayList<>();
        hashCodes = new ArrayList<>();
    }

    /**
     * Return a string containing the stats on the given game.
     * @param game The game for which statistics are required.
     * @return A list of up to 2-elements containing the stats on
     *         the players in the game, if required.
     */
    public List<PlayerStats> getStats(Game game) {
        List<PlayerStats> stats = new ArrayList<>(2);
        
        String white = game.getTagValue("White");
        String black = game.getTagValue("Black");
        String[] playerNames = { 
            white,
            black
        };
        for(String playerName : playerNames) {
            PlayerStats pstats = new PlayerStats(game, playerName, AEThreshold, showFullScores);
            if (playerMatches(playerName, playerName.equals(white)) || 
                    idMatches(pstats) ||
                    hashCodeMatches(game.getTagValue("HashCode"))) {
                double percentageWithinThreshold = pstats.getCV();
                if (scoreSettingsMatch(pstats)) {
                    stats.add(pstats);
                }
            }
        }
        return stats;
    }
    
    /**
     * Return the number of player names we are interested in.
     * @return The number of player names.
     */
    public int getNumPlayers() {
        return players.size();
    }

    /**
     * Add a player to the list of those to be matched.
     * @param player A player to be matched
     */
    public void addPlayer(String player) {
        players.add(player);
    }

    /**
     * Add the ID of a player to those to be matched.
     * @param id The ID to be matched
     */
    public void addID(String id) {
        ids.add(id);
    }

    /**
     * Return the minimum length of stats to be matched.
     * @return The minimum length
     */
    public int getMinLength() {
        return minLength;
    }

    /**
     * Set the minimum length of stats to be matched.
     * @param minLength The minimum length
     */
    public void setMinLength(int minLength) {
        this.minLength = minLength;
    }

    /**
     * Return the lower threshold of CV to be matched.
     * @return The lower threshold of CV to be matched.
     */
    public double getCVThresholdPercentage() {
        return CVThreshold;
    }

    /**
     * Set the lower threshold of CV to be matched.
     * @param percentage The lower threshold of CV to be matched.
     */
    public void setCVThresholdPercentage(double percentage) {
        this.CVThreshold = percentage;
    }

    /**
     * Return whether the full scores are to be shown.
     * @return Whether the full scores are to be shown.
     */
    public boolean isFull() {
        return showFullScores;
    }

    /**
     * Set whether the full scores are to be shown.
     * @param full Whether the full scores are to be shown.
     */
    public void setFull(boolean full) {
        this.showFullScores = full;
    }

    /**
     * @return the lowThreshold
     */
    public double getLowAEThreshold() {
        return AEThreshold;
    }

    /**
     * Set the lower threshold for the AE value.
     * @param percentage The threshold to set.
     */
    public void setLowAEThreshold(double percentage) {
        this.AEThreshold = percentage;
        thresholdSet = true;
    }

    /**
     * Return whether the accuracy is to be shown.
     * @return Whether to show the accuracy.
     */
    public boolean showAccuracy() {
        return showAccuracy;
    }

    /**
     * Set whether the accuracy is to be shown.
     * @param show Whether to show the accuracy.
     */
    public void setShowAccuracy(boolean show) {
        this.showAccuracy = show;
    }

    /**
     * Do the games settings and score settings match the criteria?
     *
     * @param stats The stats be cheched.
     * @return Whether the stats match or not.
     */
    private boolean scoreSettingsMatch(PlayerStats stats) {
        boolean matches;
        if (stats.getScores().length >= minLength) {
            if(showAccuracy) {
                matches = true;
            }
            else if (CVThreshold == 0) {
                if(thresholdSet) {
                    double mean = stats.getAE();
                    matches = mean >= AEThreshold;
                }
                else {
                    // Match anyway.
                    matches = true;
                }
            }
            else {
                // Require a percentage to match within the threshold.
                matches = stats.getCV() >= CVThreshold;
            }
            // Support random matching.
            if(!matches && randomThreshold != 0) {
                matches = rand.nextDouble() <= randomThreshold;
            }
        }
        else {
            // Too short.
            matches = false;
        }
        return matches;
    }

    /**
     * Return the random threshold setting.
     * @return The random threshold setting.
     */
    public double getRandomThreshold() {
        return randomThreshold;
    }

    /**
     * Set the random threshold.
     * @param randomThreshold The random threshold.
     */
    public void setRandomThreshold(double randomThreshold) {
        if(randomThreshold > 0 && randomThreshold <= 1.0) {
            this.randomThreshold = randomThreshold;
        }
        else {
            throw new RuntimeException("Invalid random threshold: " + randomThreshold);
        }
    }

    /**
     * Does the given player match those of interest?
     * @param player The player to check.
     * @param white Whether they played white or not.
     * @return Whether the player is of interest.
     */
    public boolean playerMatches(String player, boolean white) {
        for(String playerToMatch : players) {
            if (playerToMatch.equalsIgnoreCase(MATCH_ANY_PLAYER)) {
                return true;
            } else if (playerToMatch.equalsIgnoreCase(Game.MATCH_ANY_WHITE_PLAYER) && white) {
                return true;
            } else if (playerToMatch.equalsIgnoreCase(Game.MATCH_ANY_BLACK_PLAYER) && !white) {
                return true;
            } else if (playerToMatch.equalsIgnoreCase(player)) {
                return true;
            } else {
                // Not a match.
            }
        }
        return false;
    }
    
    /**
     * Return the current stats configuration.
     * @return The configuration as a string.
     */
    public String getConfiguration() {
        StringBuilder config = new StringBuilder();
        if(showAccuracy) {
            config.append("--accuracy ");
        }
        config.append("--AEthreshold ").append(AEThreshold).append(' ');
        config.append("--minlength ").append(minLength).append(' ');
        config.append("--CVthreshold ").append(CVThreshold).append(' ');
        return config.toString();
    }
    
    /**
     * Does the player ID in stats match one of those of interest?
     * @param stats The stats to be checked.
     * @return Whether the ID matches.
     */
    private boolean idMatches(PlayerStats stats)
    {
        String playerID = stats.buildID();
        for(String ID : ids) {
            if(ID.equals(playerID)) {
                return true;
            }
        }
        return false;
    }

    void addHashCode(String hashCode) {
        hashCodes.add(hashCode);
    }

    public boolean hashCodeMatches(String hashCodeTag) {
        if(!hashCodeTag.isEmpty()) {
            return hashCodes.contains(hashCodeTag);
        }
        else {
            return false;
        }
    }
    
}
