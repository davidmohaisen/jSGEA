/*
    gSpanCORK.cpp, v 1.2 2010/11/09 by Marisa Thoma

    is a modification based on :
    
   $Id: main.cpp,v 1.4 2004/05/21 05:50:13 taku-ku Exp $;
 
   Copyright (C) 2004 Taku Kudo, All rights reserved.
     This is free software with ABSOLUTELY NO WARRANTY.
  
   This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.
    
   This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.
    
   You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
     Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
     02111-1307, USA
     
    Modifications: removed Matlab connection, integrated nested CORK pruning
*/

#include "gspan.h"

#include <unistd.h>

#include <string>
#include <vector>

std::string gSpanCORK_executable;

void usage (void)
{
	std::cout << "gspan implementation by Taku Kudo" << std::endl;
	std::cout << "CORK feature selection included by Marisa Thoma  (compiled at " << __DATE__ << ")" << std::endl;
	std::cout << std::endl;
	std::cout << "usage: "<<gSpanCORK_executable<<" -m <minsup> [options] < GRAPHS" << std::endl;
	std::cout << std::endl;
	std::cout << "options" << std::endl;
	std::cout << "  -h:         show this usage help" << std::endl;
	std::cout << "  -m <minsup>: set the minimum support (absolute count)" << std::endl;
	std::cout << "  -t <tolerance-threshold>: when using feature selection," << std::endl;
	std::cout << "              the procedure is stopped when fewer than " << std::endl;
	std::cout << "              <tolerance-threshold> correspondences are remaining" << std::endl;
	std::cout << "  -e:         output substructures in encoded form" << std::endl;
	std::cout << "  -w <where>: if '0' no traceback of matching graphs is returned" << std::endl;
	std::cout << "              if '1' adds list of graph ids containing the frequent subgraph" << std::endl;
	std::cout << "              if '2' additionally adds these subgraphs' frequencies after a" << std::endl;
	std::cout << "                     colon: <graphId>:<frequency>" << std::endl;
	std::cout << "  -L <maxpat>: the maximum number of vertices of the selected subgraphs" << std::endl;

	std::cout << "  -n <minnodes>: the minimum number of nodes in substructes (default: 0)" << std::endl;
	std::cout << std::endl;
	std::cout << "  -x:         output in xml style" << std::endl;
	std::cout << "  -f <fs_option>: option for nested feature selection; it requires a class" << std::endl;
	std::cout << "              labelling given via the option '-l'; possible options:" << std::endl;
	std::cout << "              CORK - performs CORK selection with automatic determination"<< std::endl;
	std::cout << "                     of the number of selected subgraphs"<< std::endl;
	std::cout << "              CORK<NumSubgraphs> - performs CORK selection targeting a"<< std::endl;
	std::cout << "                     set of <NumSubgraphs> subgraphs"<< std::endl;
	std::cout << "              FCORK - faster CORK selection restricting the dataset for"<< std::endl;
	std::cout << "                     each iteration to the remaining set of unresolved"<< std::endl;
	std::cout << "                     graphs; <minsup> adapts to the remaining graph set"<< std::endl;
	std::cout << "              FCORK<NumSubgraphs> - FCORK for a limited number"<< std::endl;
	std::cout << "                     <NumSubgraphs> of subgraphs"<< std::endl;
	std::cout << "  -l <class_label_file>: file assigning row-wise class labels to graphs" << std::endl;
	std::cout << "              where row X contains the class label of graph X" <<std::endl;
	std::cout << "  -v:         also output information on the CORK statistics\n" << std::endl;
	// note that this option differs from the "verbose" parameter of the gSpan methods
	std::cout << "The graphs are read from stdin, and have to be in this format:" << std::endl;
	std::cout << "t" << std::endl;
	std::cout << "v <vertex-index> <vertex-label>" << std::endl;
	std::cout << "..." << std::endl;
	std::cout << "e <edge-from> <edge-to> <edge-label>" << std::endl;
	std::cout << "..." << std::endl;
	std::cout << "<next-graph-or-end-of-file>" << std::endl;
	std::cout << std::endl;

	std::cout << "Indices start at zero, labels are arbitrary unsigned integers." << std::endl;
	std::cout << std::endl;
}


int main (int argc, char **argv) {
  gSpanCORK_executable = std::string(argv[0]);
	unsigned int minsup = 1;
	unsigned int maxpat = 0;
	unsigned int minnodes = 2; // else causes errors
	unsigned int where = 0; // 0 => no traceback, 1 => with graph ids,
	                        // 2 => with graph ids and frequencies
	bool enc = false;
	bool directed = false;
	bool xml = false;
	bool verbose = false;

	std::string fs_option, class_label_file, correspondence_threshold;

	int opt;
	while ((opt = getopt(argc, argv, "edw:s:t:m:L:Dhn:xf:l:p:v")) != -1) {
		switch(opt) {
		case 's':
		case 'm':
			minsup = atoi (optarg);
			break;
		case 'n':
			minnodes = atoi (optarg);
			break;
		case 'L':
			maxpat = atoi (optarg);
			break;
		case 't':
			correspondence_threshold = std::string(optarg);
			break;
		case 'd': // same as original gSpan
		case 'e':
			enc = true;
			break;
		case 'w':
		  if (atoi (optarg) == 1)
		    where = 1; // traceback only
		  else if (atoi (optarg) == 2)
		    where = 2; // put out frequencies as well
			break; // else: leave at 0
		case 'D':
			directed = true;
			break;
		case 'x':
			xml = true;
			break;
		case 'f':
		  fs_option = std::string(optarg);
		  break;
		case 'l':
		  class_label_file = std::string(optarg);
		  break;
		case 'v':
			verbose = true;
			break;
		case 'h':
		default:
			usage ();
			return -1;
		}
	}

	std::set<unsigned int> graphFilter;

  if (atoi (correspondence_threshold.c_str()) > 0)
    fs_option += "C"+correspondence_threshold;

	GSPAN::gSpan gspan;
  std::ostringstream dumpStream;
  if (!verbose) { // "verbose" does not equal the "verbose" option of the "run" methods:
    // those are rather debugging statements, documenting the single algorithmic steps
    gspan.setInfoStream(dumpStream);
  }
	std::vector<GSPAN::DFSCode> selected = gspan.run_gSpan_CORK (std::cin, std::cout, minsup, minnodes, maxpat, enc, where, xml, directed, NULL, fs_option, class_label_file, NULL, false);

  return 0;
}


