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
 * Store details of a PGN tag/value pair.
 * This requires escaping some embedded characters to avoid
 * conflicts with XML characters.
 * 
 * @author David J. Barnes (d.j.barnes@kent.ac.uk)
 */
public class PGNTag {
    // The tag's name.
    private final String name;
    // The tag's value.
    private final String value;

    /**
     * Create a PGN tag from the name and value.
     * @param name The tag's name.
     * @param value The tag's value.
     */
    public PGNTag(String name, String value) {
        this.name = name;
        this.value = value;
    }

    /**
     * Return the tag's name.
     * @return The tag's name.
     */
    public String getName() {
        return name;
    }

    /**
     * Return the tag's value.
     * @return The tag's value.
     */
    public String getValue() {
        return value;
    }

    @Override
    /**
     * Recreate a PGN tag/value pair.
     * @return The tag/value pair as a PGN-style string.
     */
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("[").append(name).append(" \"");
        builder.append(XMLEscape(value));
        builder.append( "\"]");
        return builder.toString();
    }

    /**
     * Escape characters likely to be problematic in XML.
     * @param str A tag value string.
     * @return Escaped version of the value string.
     */
    private String XMLEscape(String str)
    {
        if(str.indexOf('&') >= 0 || str.indexOf('"') >= 0) {
            StringBuilder builder = new StringBuilder();
            
            int len = str.length();
            for(int index = 0; index < len; index++) {
                char ch = str.charAt(index);
                switch (ch) {
                    case '&':
                        builder.append("&amp;");
                        break;
                    case '"':
                        builder.append("&quot;");
                        break;
                    case '/':
                        if(index + 1 < len) {
                            char escaped = str.charAt(index + 1);
                            if(escaped == '&' || escaped == '"') {
                                // Add it next time around.
                            }
                            else {
                                // Leave it escaped.
                                builder.append(ch);
                                builder.append(escaped);
                                index++;
                            }
                        }
                        else {
                            // Trailing escape character.
                            builder.append(ch);
                        }   break;
                    default:
                        builder.append(ch);
                        break;
                }
            }
            return builder.toString();
        }
        else {
            return str;
        }
    }
}
