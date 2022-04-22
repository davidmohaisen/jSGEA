/*
    dfs.cpp, v 1.2 2010/11/09 by Marisa Thoma

    is a modification based on :

    $Id: dfs.cpp,v 1.3 2004/05/21 05:50:13 taku-ku Exp $;
 
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
     
    Modifications: added some format converters
*/

#include "gspan.h"

#include <cstring>
#include <assert.h>

#include <string>
#include <iterator>
#include <set>

namespace GSPAN {

/* Build a DFS code from a given graph.
 */
void
DFSCode::fromGraph (Graph &g)
{
	clear ();

	EdgeList edges;
	for (unsigned int from = 0 ; from < g.size () ; ++from) {
		if (get_forward_root (g, g[from], edges) == false)
			continue;

		for (EdgeList::iterator it = edges.begin () ; it != edges.end () ; ++it)
			push (from, (*it)->to, g[(*it)->from].label, (*it)->elabel, g[(*it)->to].label);
	}
}

/* Clear current DFS code and build code from given compressed code.
 */
void DFSCode::fromCompressedDFSCode(std::vector<int> & compressed) {
  clear ();
  
  for (unsigned int from = 0 ; from < (compressed.size()-1)/3 ; ++from) {
    std::cerr<<"error: sorry, DFSCode::fromCompressedDFSCode not implemented yet"<<std::endl;
    exit(1);
  }
}


bool DFSCode::toGraph (Graph &g) const
{
	g.clear ();

	for (DFSCode::const_iterator it = begin(); it != end(); ++it) {
		g.resize (std::max (it->from, it->to) + 1);

		if (it->fromlabel != -1) {
      assert(g[it->from].label == -1 || g[it->from].label == it->fromlabel);
			g[it->from].label = it->fromlabel;
    }
		if (it->tolabel != -1) {
      assert(g[it->to].label == -1 || g[it->to].label == it->tolabel);
			g[it->to].label = it->tolabel;
    }

		g[it->from].push (it->from, it->to, it->elabel);
		if (g.directed == false)
			g[it->to].push (it->to, it->from, it->elabel);
	}

	g.buildEdge ();

	return (true);
}

unsigned int
DFSCode::nodeCount (void) const
{
	unsigned int nodecount = 0;

	for (DFSCode::const_iterator it = begin() ; it != end() ; ++it)
		nodecount = std::max (nodecount, (unsigned int) (std::max (it->from, it->to) + 1));

	return (nodecount);
}


std::ostream &DFSCode::write (std::ostream &os)
{
	if (size() == 0) return os;

	os << "(" << (*this)[0].fromlabel << ") " << (*this)[0].elabel << " (0f" << (*this)[0].tolabel << ")";

	for (unsigned int i = 1; i < size(); ++i) {
		if ((*this)[i].from < (*this)[i].to) {
			os << " " << (*this)[i].elabel << " (" << (*this)[i].from << "f" << (*this)[i].tolabel << ")";
		} else {
			os << " " << (*this)[i].elabel << " (b" << (*this)[i].to << ")";
		}
	}

	return os;
}

/** Get a compressed version of a DFS code, which only consists
  * of integers. One DFS code starts by the vertex label of the
  * initial vertex, followed by triplets of edge specifications
  * as:
  * (elabel, from, tolabel) for forward edges,
  * (elabel, -1, to) for backward edges
  */
std::vector<int> DFSCode::getCompressedDFSCode() const {

  std::vector<int> cDFSCode(1+size()*3);
  if (size() == 0) {
    cDFSCode.clear();
    return cDFSCode;
  }
  cDFSCode[0] = at(0).fromlabel;
  
  for (unsigned int i = 0; i < size(); i++) {
    DFS dfstemp = at(i);
    if (dfstemp.from < dfstemp.to) { // forward edge
      cDFSCode[i*3+2] = dfstemp.from;
      cDFSCode[i*3+3] = dfstemp.tolabel;
    } else { // backward edge
      cDFSCode[i*3+2] = -1;
      cDFSCode[i*3+3] = dfstemp.to;
    }
    cDFSCode[i*3+1] = dfstemp.elabel;
  }
  return cDFSCode;
}
}
