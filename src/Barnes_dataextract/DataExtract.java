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

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import javax.xml.parsers.ParserConfigurationException;

/**
 * Main class for the process of extracting game data from the XML format
 * created by the game analyser.
 *
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class DataExtract {

    private int argnum;
    // Whether to annotate the game score.
    private boolean annotate;
    private boolean showStats;
    // Whether to output games that match the search criteria.
    private boolean saveMatching;
    // Whether to append to the file of matching games.
    private boolean appendToMatching;
    private boolean outputMatchingDetails;
    // Whether to output data for the win probability curve.
    private boolean outputCurveData;
    
    private String annotationFile = "annotated.txt";
    private final GenStats stats;

    /**
     * Program starting point.
     *
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        DataExtract extractor = new DataExtract();
        if (extractor.processFlags(args)) {
            if (extractor.getArgnum() < args.length) {
                extractor.processFiles(args);
            } else {
                System.err.println("Missing analysis files.");
                extractor.usage();
            }
        } else {
            extractor.usage();
        }
    }

    /**
     * Create an extractor.
     */
    private DataExtract() {
        argnum = 0;
        showStats = false;
        saveMatching = false;
        // Whether to append to the file of matching games.
        appendToMatching = false;
        outputMatchingDetails = false;
        outputCurveData = false;
        stats = new GenStats();
    }

    /**
     * Process the command-line flag settings.
     *
     * @param args Command-line arguments containing any flag settings.
     * @return true if everything was ok, false otherwise.
     */
    private boolean processFlags(String[] args) {
        boolean ok = true;
        while (argnum < args.length && args[argnum].startsWith("-")) {
            final String arg = args[argnum];
            switch (args[argnum]) {
                case "--AEthreshold":
                    argnum++;
                    if (argnum < args.length) {
                        stats.setLowAEThreshold(Double.parseDouble(args[argnum]));
                        argnum++;
                    } else {
                        System.err.println("Missing threshold after: " + arg);
                        ok = false;
                    }
                    break;
                case "--CVthreshold":
                    argnum++;
                    if (argnum < args.length) {
                        stats.setCVThresholdPercentage(Double.parseDouble(args[argnum]));
                        argnum++;
                    } else {
                        System.err.println("Missing percentage after: " + arg);
                        ok = false;
                    }
                    break;
                case "--accuracy":
                    argnum++;
                    stats.setShowAccuracy(true);
                    break;
                case "--annotate":
                    argnum++;
                    if(argnum < args.length) {
                        annotationFile = args[argnum];
                        annotate = true;
                        argnum++;
                    }
                    else {
                        System.err.println("Missing filename after: " + arg);
                        ok = false;
                    }
                    break;
                case "--append":
                    argnum++;
                    appendToMatching = true;
                    break;
                case "--curvedata":
                    argnum++;
                    outputCurveData = true;
                    break;
                case "--details":
                    argnum++;
                    outputMatchingDetails = true;
                    break;
                case "--fullstats":
                    argnum++;
                    stats.setFull(true);
                    showStats = true;
                    break;
                case "--hashfile":
                    argnum++;
                    if (argnum < args.length) {
                        ok = addHashCodesFromFile(args[argnum]);
                        argnum++;
                    } else {
                        System.err.println("Missing filename after: " + arg);
                        ok = false;
                    }
                    break;
                case "--help":
                    argnum++;
                    usage();
                    System.exit(0);
                    break;
                case "--id":
                    argnum++;
                    if (argnum < args.length) {
                        stats.addID(args[argnum]);
                        argnum++;
                    } else {
                        System.err.println("Missing id string after: " + arg);
                        ok = false;
                    }
                    break;
                case "--idfile":
                    argnum++;
                    if (argnum < args.length) {
                        ok = addIDsFromFile(args[argnum]);
                        argnum++;
                    } else {
                        System.err.println("Missing filename after: " + arg);
                        ok = false;
                    }
                    break;
                case "--matching":
                    argnum++;
                    saveMatching = true;
                    break;
                case "--minlength":
                    argnum++;
                    if (argnum < args.length) {
                        stats.setMinLength(Integer.parseInt(args[argnum]));
                        argnum++;
                    } else {
                        System.err.println("Missing non-book game length after: " + arg);
                        ok = false;
                    }
                    break;
                case "--player":
                    argnum++;
                    if (argnum < args.length) {
                        stats.addPlayer(args[argnum]);
                        argnum++;
                    } else {
                        System.err.println("Missing player name after: " + arg);
                        ok = false;
                    }
                    break;
                case "--random":
                    argnum++;
                    if (argnum < args.length) {
                        stats.setRandomThreshold(Double.parseDouble(args[argnum]));
                        argnum++;
                    } else {
                        System.err.println("Missing threshold after: " + arg);
                        ok = false;
                    }
                    break;
                case "--stats":
                    argnum++;
                    showStats = true;
                    break;
                default:
                    argnum++;
                    System.err.println("Unrecognised argument: " + arg);
                    ok = false;
                    break;
            }
        }
        return ok;
    }

    /**
     * Process any files containing analysed games.
     *
     * @param args Command-line arguments containing file names.
     */
    private void processFiles(String[] args) {
        XMLProcessor processor = new XMLProcessor();
        if(!annotate) {
            // Show the configuration.
            System.out.println("# " + stats.getConfiguration());
            System.out.println("# Date:Player:W/B:BD:EM:Depth:AE:sd:CV:Res:Hash:");
        }
        try {
            FileWriter gameFile = null;
            FileWriter detailsFile = null;
            FileWriter annotatedFile = null;
            
            if (annotate) {
                annotatedFile = new FileWriter(annotationFile, appendToMatching);
            }
            if (outputMatchingDetails) {
                detailsFile = new FileWriter("details.txt", appendToMatching);
            }
            if (saveMatching) {
                gameFile = new FileWriter("matching.pgn", appendToMatching);
            }

            List<Game> gameList = null;
            while (argnum < args.length) {
                try {
                    processor.processXMLFile(args[argnum]);
                    gameList = processor.getGameList();
                    // Show the stats and save the matching games.
                    for (Game game : gameList) {
                        if (annotate) {
                            game.annotate(annotatedFile);
                            annotatedFile.write('\n');
                        }
                        else if(outputCurveData) {
                            game.outputCurveData();
                            if(saveMatching && stats.hashCodeMatches(game.getTagValue("HashCode"))) {
                                gameFile.write(game.toString());
                                gameFile.write("\n");                                
                            }
                        } else {
                            List<PlayerStats> ps = stats.getStats(game);
                            for (PlayerStats s : ps) {
                                System.out.println(s);
                                if (outputMatchingDetails) {
                                    detailsFile.write(game.getAnalysis().toString());
                                    detailsFile.write("\n");
                                }
                                if (saveMatching) {
                                    gameFile.write(game.toString());
                                    gameFile.write("\n");
                                }
                            }
                        }
                    }
                } catch (FileNotFoundException ex) {
                    System.err.println("File not found: " + args[argnum]);
                    System.exit(1);
                } catch (IOException ex) {
                    System.err.println("Error processing: " + args[argnum]);
                }
                // The game list is shared between files so it must be cleared
                // each time.
                if(gameList != null) {
                    gameList.clear();
                }
                argnum++;
            }
            if (annotatedFile != null) {
                annotatedFile.close();
            }
            if (gameFile != null) {
                gameFile.close();
            }
            if (detailsFile != null) {
                detailsFile.close();
            }
        } catch (ParserConfigurationException ex) {
            System.err.println("Parser configuration: " + args[argnum]);
        } catch (IOException ex) {
            System.err.println("Fatal IO error.");
        }
    }
    
    /**
     * Print a usage message to standard output.
     */
    private void usage() {
        System.out.println("Usage: "
                + "[--AEthreshold D] "
                + "[--CVthreshold D] "
                + "[--fullstats] "
                + "[--help] "
                + "[--id id-string] "
                + "[--idfile filename] "
                + "[--matching] "
                + "[--minlength N] "
                + "[--player name] "
                + "[--random probability] "
                + "[--stats] "
                + " file ...");
    }

    /**
     * Add from the given file any game IDs to be matched.
     *
     * @param filename The file of IDs.
     * @return true if IDs added ok, false otherwise.
     */
    private boolean addIDsFromFile(String filename) {
        boolean ok;
        try (BufferedReader reader = new BufferedReader(
                        new FileReader(filename))) {
            String id = reader.readLine();
            while (id != null) {
                id = id.trim();
                if (!id.isEmpty()) {
                    stats.addID(id);
                }
                id = reader.readLine();
            }
            ok = true;
        } catch (IOException e) {
            System.err.println("Error reading: " + filename);
            ok = false;
        }
        return ok;
    }

    /**
     * Add from the given file any game hashcodes to be matched.
     *
     * @param filename The file of IDs.
     * @return true if IDs added ok, false otherwise.
     */
    private boolean addHashCodesFromFile(String filename) {
        boolean ok;
        try (BufferedReader reader = new BufferedReader(
                        new FileReader(filename))) {
            String hashCode = reader.readLine();
            while (hashCode != null) {
                hashCode = hashCode.trim();
                if (!hashCode.isEmpty()) {
                    stats.addHashCode(hashCode);
                }
                hashCode = reader.readLine();
            }
            ok = true;
        } catch (IOException e) {
            System.err.println("Error reading: " + filename);
            ok = false;
        }
        return ok;
    }

    /**
     * Return the current value of argnum - the number of arguments processed.
     *
     * @return The value of argnum.
     */
    private int getArgnum() {
        return argnum;
    }
}
