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
 * Store details of the score for a played move.
 * 
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class PlayedMoveScore {
    // Is the best move forced mate?
    private boolean bestIsMate;
    // false unless bestIfMate, in which case it will be
    // true unless the move played also led to forced mate.
    private boolean playedIsMate; 
    // Difference between best and played evaluations, unless
    // bestIsMate && !playedIsMate, in which case it is the
    // difference between mate length.
    // value is negative when !bestIsMate
    // value is >= 0 when bestIsMate && !playedIsMate.
    // value is the score of the played move when bestIsMate && playedIsMate.
    private int value; 
    
    /**
     * The value of a move.
     * @param value The move's value.
     */
    public PlayedMoveScore(int value)
    {
        this(value, false, false);
    }
    
    /**
     * The value of a move.
     * @param value The move's value.
     * @param bestIsMate Whether the best move was mate.
     * @param playedIsMate Whether the played move was mate.
     */
    public PlayedMoveScore(int value, boolean bestIsMate, boolean playedIsMate)
    {
        this.value = value;
        this.bestIsMate = bestIsMate;
        this.playedIsMate = playedIsMate;
    }

    /**
     * Is the best move a mating move?
     * @return true if the best move is a mating move.
     */
    public boolean bestIsMate() {
        return bestIsMate;
    }

    /**
     * Is the move played a mating move?
     * @return true if the played move is a mating move.
     */
    public boolean playedIsMate() {
        return playedIsMate;
    }

    /**
     * Difference between best and played evaluations, unless
     * bestIsMate && !playedIsMate, in which case it is the
     * difference between mate length.
     * value is negative when !bestIsMate
     * value is >= 0 when bestIsMate && !playedIsMate.
     * value is the score of the played move when bestIsMate && playedIsMate.
     * @return 
     */
    public int getValue() {
        return value;
    }
    
    @Override
    public String toString()
    {
        return value + " bm: " + bestIsMate + " pm: " + playedIsMate;
    }
}
