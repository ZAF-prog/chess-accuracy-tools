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

/**
 * Provide details of a particular player in a particular game.
 * 
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class PlayerStats {
    // Used for formatting player names.
    private static final int MAX_NAME_LENGTH = 30;
    
    // The game.
    private final Game game;
    // Whether the player was white or black.
    private final boolean isWhite;
    // The low threshold for returning scores.
    private final double lowThreshold;
    // Whether full scores should be shown.
    private final boolean showFullScores;
    // The name of the player.
    private final String playerName;
    // The scores for each move.
    private final int[] scores;
    // Textual version of the score differences for each move.
    private final String[] textDifferences;
    // Overall coincidence value of the moves.
    private final double CV;
    
    /**
     * Record stats for a player in a game.
     * @param game The game.
     * @param playerName The player.
     * @param lowThreshold The low threshold for scores.
     * @param showFullScores Whether to show full scores or not.
     */
    public PlayerStats(Game game, String playerName, double lowThreshold, boolean showFullScores) {
        this.game = game;
        this.playerName = playerName;
        this.isWhite = game.getTagValue("White").equalsIgnoreCase(playerName);
        this.lowThreshold = lowThreshold;
        this.showFullScores = showFullScores;
        
        scores = game.getNonMateScores(playerName);
        textDifferences = game.getScoresAsText(playerName);
        CV = getPercentageWithinThreshold(scores, lowThreshold);
    }

    @Override
    public String toString()
    {
        StringBuilder output = new StringBuilder();
       // Output the date and differences on a single line.
        String date = game.getTagValue("Date");
        if (date.length() < 4) {
            date = "????";
        }
        String result = game.getTagValue("Result");
        if(result.length() >= 3) {
            result = result.substring(0, 3);
        }
        else {
            result = String.format("%3s", result);
        }
        String hashcode = game.getTagValue("HashCode");
        while(hashcode.length() < 8) {
            hashcode = "0" + hashcode;
        }
        output.append(date.substring(0, 4));
        output.append(':');
        // Print the player's name.
        output.append(String.format("%-30s", 
                playerName.substring(0, Math.min(MAX_NAME_LENGTH, playerName.length()))));
        if (isWhite) {
            output.append(":W");
        } else  {
            output.append(":B");
        }
        double mean = getAE();
        double sd = sd(mean);
        // Output textDifferences.length rather than scores.length to avoid
        // inconsistencies with non-numeric mate scores.
        output.append(':');
        output.append(String.format("%3s", game.getAnalysis().getBookDepth()));
        output.append(':');
        output.append(String.format("%3d", textDifferences.length));
        output.append(':');
        output.append(String.format("%2s", game.getAnalysis().getSearchDepth()));
        output.append(':');
        output.append(String.format("%8.2f", mean)).append(':');
        output.append(String.format("%6.1f", sd)).append(':');
        output.append(String.format("%5.2f", CV)).append(':');
        
        output.append(result).append(':');
        output.append(hashcode).append(':');
        if (showFullScores) {
            // Print the raw scores, too.
            for (String score : textDifferences) {
                output.append(String.format("%s:", score));
            }
        }
        return output.toString();
    }

    /**
     * Return the scores.
     * @return The scores.
     */
    public int[] getScores() {
        return scores;
    }
    
    /**
     * Return the number of scores for this player.
     * @return The number of scores.
     */
    public int getNumScores()
    {
        return scores.length;
    }

    /**
     * Return a text version of the score differences.
     * @return The score differences.
     */
    public String[] getTextDifferences() {
        return textDifferences;
    }

    /**
     * Return the percentage of scores that are within
     * the threshold.
     * The Coincidence Value (CV).
     * @return The coincidence value.
     */
    public double getCV() {
        return CV;
    }
    
    /**
     * Return the percentage of scores that are within the threshold.
     * @param scores
     * @return 
     */
    private double getPercentageWithinThreshold(int[] scores, double lowThreshold) {
        int withinCount = 0;
        for(int score : scores) {
            if(score >= lowThreshold) {
                withinCount++;
            }
        }
        return ((double) withinCount) / scores.length;
    }
    
    /**
     * Return the mean score.
     * The Average Error (AE).
     *
     * @return
     */
    public double getAE() {
        int sum = 0;
        for (int score : scores) {
            sum += score;
        }
        return ((double) sum) / scores.length;
    }

    /**
     * Return the standard deviation of the scores.
     *
     * @param The mean of the scores.
     * @return The standard deviation.
     */
    private double sd(double mean) {
        double sum = 0;
        for (int score : scores) {
            double diff = mean - score;
            sum += diff * diff;
        }
        return Math.sqrt(sum / scores.length);
    }

    /**
     * Build the ID string for the given player on the given game.
     * @return The ID string.
     */
    public String buildID()
    {
        StringBuilder builder = new StringBuilder();
        String date = game.getTagValue("Date");
        if (date.length() < 4) {
            date = "????";
        }
        builder.append(date.substring(0, 4));
        builder.append(':');
        builder.append(String.format("%-30s", 
                playerName.substring(0, Math.min(MAX_NAME_LENGTH, 
                playerName.length()))));
        if(isWhite) {
            builder.append(":W:");
        }
        else {
            builder.append(":B:");
        }
        builder.append(String.format("%3s", game.getTagValue("BookDepth")));
        builder.append(':');
        builder.append(String.format("%3d", game.getScoresAsText(playerName).length));
        return builder.toString();
    }

}
