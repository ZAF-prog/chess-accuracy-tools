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
 * Capture the evaluation of a single move.
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class Evaluation {
    // The move.
    private final String move;
    // The move's evaluation.
    private final String evaluation;

    /**
     * Store details of the evaluation of a move.
     * @param move The move.
     * @param evaluation The move's evaluation.
     */
    public Evaluation(String move, String evaluation) {
        this.move = move;
        this.evaluation = evaluation;
    }

    /**
     * Return the move.
     * @return The move.
     */
    public String getMove() {
        return move;
    }

    /**
     * Return the evaluation of the move.
     * @return The evaluation.
     */
    public String getEvaluation() {
        return evaluation;
    }
    
    @Override
    /**
     * @return The move and its evaluation.
     */
    public String toString()
    {
        return move + "  " + evaluation;
    }
}
