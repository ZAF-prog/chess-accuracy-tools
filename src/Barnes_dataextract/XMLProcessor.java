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

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

/**
 * XML processor the files output by the analyser.
 *
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class XMLProcessor {

    private static final String 
            ANALYSIS_TAG = "analysis",
            EVALUATION_TAG = "evaluation",
            GAME_TAG = "game",
            GAMELIST_TAG = "gamelist",
            MOVE_TAG = "move",
            MOVES_TAG = "moves",
            PLAYED_TAG = "played",
            TAG_TAG = "tag",
            TAGS_TAG = "tags";
    private static final String SEARCH_DEPTH = "searchDepth";
    private static final String BOOK_DEPTH = "bookDepth";
    private static final String ENGINE = "engine";
    // The list of nodes.
    private NodeList nodeList;
    // The list of games in the file.
    private final List<Game> gameList;

    /**
     * Create a processor.
     */
    public XMLProcessor() {
        nodeList = null;
        gameList = new ArrayList<>();
    }

    /**
     * Process the given XML file.
     *
     * @param filename The name of the file.
     * @throws ParserConfigurationException on configuration error.
     * @throws IOException on file-processing errors.
     */
    public void processXMLFile(String filename) throws ParserConfigurationException, IOException {
        File file = new File(filename);
        // Configure the input source for UTF-8 encoding, just in case.
        try (Reader reader = new InputStreamReader(new FileInputStream(file), "UTF-8")) {
            InputSource utf8Source = new InputSource(reader);
            utf8Source.setEncoding("UTF-8");

            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(utf8Source);
            doc.getDocumentElement().normalize();
            nodeList = doc.getElementsByTagName(GAMELIST_TAG);
            processGameList(nodeList, 0);
        } catch (SAXException e) {
            System.err.println("Sax exception in " + filename + ": " + e);
        } catch (IOException e) {
            System.err.println("IOException in " + filename + ": " + e);
        }
    }

    /**
     * Return the list of games extracted.
     *
     * @return The list of games.
     */
    public List<Game> getGameList() {
        return gameList;
    }

    /**
     * Process a list of nodes representing games.
     *
     * @param list The list of nodes.
     * @param depth The current node depth.
     */
    private void processGameList(NodeList list, int depth) {
        List<PGNTag> tagList = null;
        String moves = null;
        int listLength = list.getLength();
        for (int n = 0; n < listLength; n++) {
            Node node = list.item(n);
            String value = node.getNodeValue();

            switch (node.getNodeType()) {
                case Node.ELEMENT_NODE:
                    Element element = (Element) node;
                    String eName = element.getNodeName();
                    switch (eName) {
                        case ANALYSIS_TAG:
                            Analysis analysis = process_analysis_tag(node.getChildNodes());
                            NamedNodeMap map = node.getAttributes();
                            int mapLength = map.getLength();
                            for (int index = 0; index < mapLength; index++) {
                                String nodeName = map.item(index).getNodeName();
                                if (nodeName.equalsIgnoreCase(SEARCH_DEPTH)) {
                                    analysis.setSearchDepth(map.item(index).getNodeValue());
                                } else if (nodeName.equalsIgnoreCase(BOOK_DEPTH)) {
                                    analysis.setBookDepth(map.item(index).getNodeValue());
                                } else if (nodeName.equalsIgnoreCase(ENGINE)) {
                                    analysis.setEngineID(map.item(index).getNodeValue());
                                }
                            }
                            gameList.add(new Game(tagList, moves, analysis));
                            break;
                        case MOVES_TAG:
                            // Get the moves as the text of the single child node.
                            if (node.hasChildNodes()) {
                                NodeList children = node.getChildNodes();
                                moves = children.item(0).getNodeValue();
                            } else {
                                moves = "??";
                            }
                            break;
                        case TAGS_TAG:
                            tagList = process_tag_list(node.getChildNodes());
                            break;
                        default:
                            break;
                    }
                    break;
            }
            if (node.hasChildNodes()) {
                processGameList(node.getChildNodes(), depth + 1);
            }
        }
    }

    /**
     * Process a list of tags
     *
     * @param childNodes Child nodes for the tags.
     * @return The list of tags.
     */
    private List<PGNTag> process_tag_list(NodeList childNodes) {
        List<PGNTag> tagList = new ArrayList<>();
        int listLength = childNodes.getLength();
        for (int n = 0; n < listLength; n++) {
            Node node = childNodes.item(n);
            String value = node.getNodeValue();
            if (node.getNodeType() == Node.ELEMENT_NODE
                    && node.getNodeName().equals(TAG_TAG)) {
                tagList.add(process_tag_tag(node));
            }
        }
        return tagList;

    }

    /**
     * Process a PGNTag node.
     *
     * @param node The node for the PGN tag.
     * @return A PGNTag.
     */
    private PGNTag process_tag_tag(Node node) {
        NamedNodeMap map = node.getAttributes();
        Node eNode = map.item(0);
        return new PGNTag(map.item(0).getNodeValue(), map.item(1).getNodeValue());
    }

    /**
     * Process the analysis node which contains the analysis of all the moves.
     *
     * @param childNodes The nodes for the analysis.
     * @return The analysis.
     */
    private Analysis process_analysis_tag(NodeList childNodes) {
        // Assume a standard game, with white to move first.
        // NB: from version 2017.04.05 of the analyser, this is unnecessary
        // because who is making the move is encoded in the <move> element.
        boolean whiteToMove = true;
        Analysis analysis = new Analysis();
        int listLength = childNodes.getLength();
        for (int n = 0; n < listLength; n++) {
            Node node = childNodes.item(n);
            String value = node.getNodeValue();
            if (node.getNodeType() == Node.ELEMENT_NODE) {
                if (node.getNodeName().equals(MOVE_TAG)) {
                    // Look for player = "white" or player = "black" as an attribute.
                    NamedNodeMap map = node.getAttributes();
                    int mapLength = map.getLength();
                    for (int index = 0; index < mapLength; index++) {
                        Node attrNode = map.item(index);
                        String attrName = attrNode.getNodeName();
                        if (attrName.equalsIgnoreCase("player")) {
                            String attrValue = attrNode.getNodeValue();
                            if(attrValue.equalsIgnoreCase("white")) {
                                whiteToMove = true;
                            }
                            else if(attrValue.equalsIgnoreCase("black")) {
                                whiteToMove = false;
                            }
                        }
                    }
                    analysis.addAnalysedMove(process_move_tag(node.getChildNodes(), whiteToMove));
                    // Legacy.
                    whiteToMove = !whiteToMove;
                }
            }
        }
        return analysis;
    }

    /**
     * Process the nodes for a move.
     *
     * @param childNodes The nodes of the move.
     * @param whiteMove Whether it is white's move.
     * @return The played move and its evaluations.
     */
    private PlayedMove process_move_tag(NodeList childNodes, boolean whiteMove) {
        PlayedMove move = null;
        int listLength = childNodes.getLength();
        for (int n = 0; n < listLength; n++) {
            Node node = childNodes.item(n);
            String nodeName = node.getNodeName();
            String value = node.getNodeValue();
            switch (nodeName) {
                case PLAYED_TAG: {
                    String moveText = "???";
                    if (node.hasChildNodes()) {
                        NodeList children = node.getChildNodes();
                        if (children.getLength() == 1) {
                            moveText = children.item(0).getNodeValue();
                        }
                    }
                    move = new PlayedMove(moveText, whiteMove);
                }
                break;
                case EVALUATION_TAG: {
                    NamedNodeMap map = node.getAttributes();
                    int mapLength = map.getLength();
                    if (mapLength == 2) {
                        String moveText = map.item(0).getNodeValue();
                        String evaluation = map.item(1).getNodeValue();
                        if(move != null) {
                            move.addEvaluation(new Evaluation(moveText, evaluation));
                        }
                        else {
                            System.err.println("No played move found for evaluation of " + moveText);
                        }
                    }
                }
                break;
                default:
                    break;
            }
        }
        return move;
    }
}
