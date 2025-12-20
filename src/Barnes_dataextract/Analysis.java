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

/**
 * Store details of the moves analysed in a single game.
 * This includes the search depth and the book depth.
 * 
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class Analysis {
    // The moves that were analysed.
    private final List<PlayedMove> analysedMoves;
    // The id of the engine.
    private String engineID;
    // The engine's search depth.
    private String searchDepth;
    // The book depth of the game. These moves will not
    // have been analysed.
    private int bookDepth;

    public Analysis() {
        this.analysedMoves = new ArrayList<>();
        this.engineID = "unknown";
        this.searchDepth = "";
        this.bookDepth = -1;
    }
    
    /**
     * Add a move to the list of those analysed.
     * @param move The analysed move.
     */
    public void addAnalysedMove(PlayedMove move)
    {
        analysedMoves.add(move);
    }

    /**
     * Return the analysed moves.
     * @return The analysed moves.
     */
    public List<PlayedMove> getAnalysedMoves() {
        return analysedMoves;
    }

    /**
     * Return the search depth used with this game.
     * @return The search depth.
     */
    public String getSearchDepth() {
        return searchDepth;
    }

    /**
     * Set the search depth for this game.
     * @param searchDepth The search depth.
     */
    public void setSearchDepth(String searchDepth) {
        this.searchDepth = searchDepth;
    }

    /**
     * Get the ID of the engine.
     * @return The engine ID.
     */
    public String getEngineID() {
        return engineID;
    }

    /**
     * Set the ID of the engine.
     * @param engineID The engine ID.
     */
    public void setEngineID(String engineID) {
        this.engineID = engineID;
    }
    
    @Override
    /**
     * Return the played moves as a string.
     */
    public String toString()
    {
        StringBuilder builder = new StringBuilder();
        for(PlayedMove move : analysedMoves) {
            builder.append(move).append('\n');
        }
        return builder.toString();
    }

    /**
     * Return the book depth of this game.
     * @return The book depth.
     */
    public int getBookDepth() {
        return bookDepth;
    }

    /**
     * Set the book depth of this game.
     * @param nodeValue The node containing the book depth.
     */
    public void setBookDepth(String nodeValue) {
        bookDepth = Integer.parseInt(nodeValue);
        if(bookDepth < 0) {
            bookDepth = -1;
        }
    }
}
