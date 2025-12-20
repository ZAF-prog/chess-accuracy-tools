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
import java.util.Iterator;
import java.util.List;

/**
 * Store details of a played move and its evaluations.
 * 
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class PlayedMove {
    // Whether the move is for white (true) or black (false).
    private final boolean whiteMove;
    // The move that was played.
    private final String move;
    // Evaluations for the move and its alternatives.
    private final List<Evaluation> evaluations;

    /**
     * Record details for a move.
     * @param move The move.
     * @param whiteMove Whether white is to move.
     */
    public PlayedMove(String move, boolean whiteMove) {
        this.whiteMove = whiteMove;
        this.move = move;
        evaluations = new ArrayList<>();
    }
    
    /**
     * Add an evaluation for this move.
     * @param eval The evaluation.
     */
    public void addEvaluation(Evaluation eval)
    {
        evaluations.add(eval);
    }
    
    /**
     * Return the evaluation for the move played.
     * @return Return the evaluation, or null if it is not found.
     */
    public Evaluation getEvaluationForMove()
    {
        // Return the evaluation for the given move.
        Evaluation ev = null;
        Iterator<Evaluation> it = evaluations.iterator();
        while(ev == null && it.hasNext()) {
            Evaluation eval = it.next();
            if(move.equals(eval.getMove())) {
                ev = eval;
            }
        }
        return ev;
    }
    
    /**
     * Return the first evaluation for this move.
     * @return 
     */
    public Evaluation getFirstEvaluation()
    {
        return evaluations.get(0);
    }

    /**
     * Whether the move was played by white (true) or black (false)
     * @return true for a white move, false otherwise.
     */
    public boolean isWhiteMove() {
        return whiteMove;
    }

    /**
     * Return the move.
     * @return The move.
     */
    public String getMove() {
        return move;
    }

    /**
     * Return the evaluations for this move.
     * @return The evaluations.
     */
    public List<Evaluation> getEvaluations() {
        return evaluations;
    }
    
    @Override
    /**
     * Return the move and its evaluations.
     * @return The move and its evaluations.
     */
    public String toString()
    {
        StringBuilder builder = new StringBuilder();
        builder.append(move).append('\n');
        for(Evaluation ev : evaluations) {
            builder.append("  ");
            builder.append(ev);
            builder.append('\n');
        }
        return builder.toString();
    }
    
    
}
