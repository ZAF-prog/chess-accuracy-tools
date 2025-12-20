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

This directory contains the source code for an XML processor designed to
process the output from the UCI-based analyser which is available from:
https://www.cs.kent.ac.uk/~djb/chessplag/

Overview
========
The purpose of the analyser is to read a source file containing moves of
one or more chess games and pass them to a UCI-compatible chess engine.
The analyser receives back the engine's evaluations and writes them out in
an XML format for processing by another program. The XML processor is designed
to read that XML output and provide summary statistical information from it.

Execution
=========
The program is provided as a JAR file, dataextract.jar, which can be executed
as:

    java -jar dataextract.jar

given a suitable installation of the Java Runtime Environment (JRE).

Usage
=====
The analyser takes 0 or more command-line options (see below) and 1 or more
XML files.

    java -jar dataextract.jar [optional-command-line-options] xml-file ...

Summary
=======
In essence, the program is used to provide summary statistics on the degree to
which the moves played in a chess game match those selected by a UCI-compatible
chess engine. The two key statistics are CV and AE. These are defined as follows:

+ Coincidence Value (CV) is a figure between 0 and 1 representing
the proportion of non-book moves chosen by a player with the same evaluation as
the engine's preferred move.

+ Average error (AE) is the mean difference in evaluation between the best move
and the played move for non-book moves, expressed in centipawns.

For each game analysed, two lines are output (one for each player) in the following
format:

Date:Player:W/B:BD:EM:Depth:AE:sd:CV:Res:Hash

Fields are separated with a colon character. The fields have a fixed-width, as indicated:

+ Date (4): The year of the game.
+ Player (30): The player's name.
+ W/B (1): Either W or B depending upon which player colour the data refers to.
+ BD (3): The book depth. The number of ply considered book and, therefore, not analysed.
+ EM (3): The number of moves evaluated for this player in this game.
+ Depth (2): The UCI engine's search depth.
+ AE (8): The AE value for this player's moves in this game.
+ sd (6): The standard deviation of the score differences for this player in this game.
+ CV (5): The CV value for this player's moves in this game.
+ Res (3): The result of this game 1-0, 0-1, 1/2
+ Hash (8): A hash of the game's moves.

For instance:

2002:Pyshkin, Aleksandr Sergeevic  :B:  8: 22:14:  -19.27:  32.9: 0.41:

This line indicates a game played in 2002 where Pyshkin played black. The first 8
ply were treated as book moves and 22 further moves by Pyshkin were analysed to depth
14. The average centipawn difference in evaluation between the moves played by Pyshkin
and the engine's assessment of the best move was -19.27. The standard deviation was
32.9. 41% of the moves played by Pyshkin were considered equal in evaluation to the
best move at that point by the engine.  

Command-line Options
====================
--AEthreshold D - set the lower AE threshold for outputting details of games to D.
--CVthreshold D - set the lower CV threshold for outputting details of games to D (0-1.0).
--fullstats - output the differences values of each move.
--help - show the usage information.
--id id-string - output only games with the given ID (see below for ID).
--idfile filename - output only games with the IDs listed in filename.
--matching - output the PGN for games that are output, in the file matching.pgn
--minlength N - only output games with a minimum of N evaluated moves.
--player name - only output games played by the given player. NB <White>, <Black> and
<WhiteOrBlack> will match any player playing white, black, or either colour, respectively.
--random probability - randomly select games to be output with the given probability (0-1.0).
--stats - output stats on the game to standard output (default).

ID strings
==========
The --id option allows selection of a game with specific details. The --idfile option allows
multiple ID strings to be stored in a file to simplify command-line usage.

An ID consists of the first 5 fields of the output format:

Date:Player:W/B:BD:EM

where Date is 4 characters, Player is 30 characters, W/B is 1 character, BD is two characters
and EM is 3 characters.

Examples
========
  java -jar dataextract.jar --player "<WhiteOrBlack>" file.xml

will output the stats for all players in the games found in file.xml

  java -jar dataextract.jar --player "<White>" --matching file.xml

will output the stats for all players playing White and write a file, matching.pgn,
containing the games.

  java -jar dataextract.jar --player "Morphy, Paul" --CVthreshold 0.9 file.xml

will output only those statistics for games in file.xml in which Paul Morphy's moves have
a CV value of at least 90%.

  java -jar dataextract.jar --id "2002:Pyshkin, Aleksandr Sergeevic  :B:  8: 22" --matching file.xml

will output the game described above that was played in 2002 by Pyshkin.

David J. Barnes
30th June 2014
Updated: 2017.04.05
