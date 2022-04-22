/*
    gspan.cpp, v 1.2 2010/11/09 by Marisa Thoma

    is a modification based on :

    $Id: gspan.cpp,v 1.8 2004/05/21 09:27:17 taku-ku Exp $;
 
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

   Mmodified by Marisa Thoma (extracted from matlab; extended to CORK pruning)
     
*/
#include "gspan.h"

#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <limits>

#include <iterator>
#include <iostream>
#include <cassert>

namespace GSPAN {

gSpan::gSpan (void) { infoStream = &std::cerr; }

std::istream &gSpan::read (std::istream &is,
    std::set<unsigned int> const * filter) {

  Graph g(directed);
  std::string gName;
  unsigned int gNameInt;
  while (true) {
    g.read (is, &gName);
    gNameInt = strtoul(gName.c_str(),NULL,10);
    if (g.empty()) break;
    if (filter != NULL) { // select for elements in filter
      if (filter->find(gNameInt) == filter->end()) // graph not selected
        continue;
    }
    TRANS.push_back (g);
    // assign class labels if possible
    if (instances2classLabels.size() != 0) {
      std::map<unsigned int, unsigned int>::const_iterator clIter = instances2classLabels.find(gNameInt);
      if (clIter == instances2classLabels.end()) {
      	std::cerr <<"error: instance \""<<gNameInt<<"\" is not covered in class label map (gSpan::read:"<<__LINE__<<")"<<std::endl;
      	exit(1);
      }
      class_labels.push_back(clIter->second);
      originalInstanceLabels.push_back(gNameInt);
    }
  }

  return is;
}

std::map<unsigned int, unsigned int> gSpan::support_counts (Projected &projected) {
	std::map<unsigned int, unsigned int> counts;

	for (Projected::iterator cur = projected.begin() ;
		cur != projected.end() ; ++cur)
	{
	  counts[cur->id] ++;
	}
	return counts;
}

unsigned int gSpan::support (Projected &projected) {
	unsigned int oid = 0xffffffff;
	unsigned int size = 0;

	for (Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
		if (oid != cur->id) {
			++size;
		}
		oid = cur->id;
	}

	return size;
}

/* Special report function for single node graphs. -- NOT supported by FS
 */
void gSpan::report_single (Graph &g,
    std::map<unsigned int, unsigned int>& ncount) {
	unsigned int sup = 0;
	for (std::map<unsigned int, unsigned int>::iterator it = ncount.begin () ;
       it != ncount.end () ; ++it) {
		sup += (*it).second;
	}

	if (maxpat_max > maxpat_min && g.size () > maxpat_max)
		return;
	if (maxpat_min > 0 && g.size () < maxpat_min)
		return;
	if (enc == false) {
	  *os << "t # " << ID << " * " << sup;
	  *os << '\n';
	  
	  g.write (*os);
	  *os << '\n';
	} else {
	  std::cerr << "report_single not implemented for non-Matlab calls" << std::endl;
	}
	
}

void gSpan::report (Projected &projected, unsigned int sup) {

	/* Filter too small / too large graphs.
	 */
	if (maxpat_max > maxpat_min && DFS_CODE.nodeCount () > maxpat_max)
		return;
	if (maxpat_min > 0 && DFS_CODE.nodeCount () < maxpat_min)
		return;

	if (xml && where) {
		*os << "<pattern>\n";
		*os << "<id>" << ID << "</id>\n";
		*os << "<support>" << sup << "</support>\n";
		*os << "<what>";
	}

	if (! enc) {
		Graph g(directed);
		DFS_CODE.toGraph (g);

		if (! xml && where) // separate
		  *os << std::endl;

		*os << "t # " << ID << " * " << sup;
		*os << '\n';
		g.write (*os);
	} else {
		if (! xml || ! where)
			*os << '<' << ID << ">    " << sup << " [";

		DFS_CODE.write (*os);
		if (! xml || ! where)
      *os << ']';
	}

	if ((bool)where) { // list graph ids for pattern
	  if (xml) // close tag
	    *os << "</what>\n<where>";
	  else
	    *os << " {";

	  unsigned int oid = 0xffffffff;
	  if (where == 1) { // => only report graph ids as traceback
	    for (Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
	      if (oid != cur->id) {
      		if (cur != projected.begin())
            *os << ' ';
      		if (originalInstanceLabels.size() == 0)
      		  *os << cur->id;
      		else
      		  *os << originalInstanceLabels.at(cur->id);
	      }
	      oid = cur->id;
	    }
	  } else { // must be 2 => also report hit frequencies
	    unsigned int freqCount = 0;
	    for (Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
	      if (oid != cur->id) {
      		if (cur != projected.begin()) { // freqCount == 0
      		  *os << ':' << freqCount << ' ';
      		  freqCount = 0;
      		}
      		if (originalInstanceLabels.size() == 0)
      		  *os << cur->id;
      		else
      		  *os << originalInstanceLabels.at(cur->id);

      		oid = cur->id;
	      }
	      freqCount++;
	    }
	    *os << ':' << freqCount; // last frequency
	  }
	  if (xml)
	    *os << "</where>\n</pattern>";
	  else
	    *os << '}';
	}

	*os << '\n';
	++ID;
}

void gSpan::report (DFSCode & toReport,
    std::map<unsigned int, unsigned int> & supp_counts) {

	/* Filter too small / too large graphs.
	 */
	if (maxpat_max > maxpat_min && toReport.nodeCount () > maxpat_max)
		return;
	if (maxpat_min > 0 && toReport.nodeCount () < maxpat_min)
		return;

	if (xml && where) {
		*os << "<pattern>\n";
		*os << "<id>" << ID << "</id>\n";
		*os << "<support>" << supp_counts.size() << "</support>\n";
		*os << "<what>";
	}

	if (! enc) {
		Graph g(directed);
		toReport.toGraph (g);

		if (! xml && where) // separate
		  *os << std::endl;

		*os << "t # " << ID << " * " << supp_counts.size();
		*os << std::endl;
		g.write (*os);
	} else {
		if (! xml || ! where)
			*os << '<' << ID << ">    " << supp_counts.size() << " [";

		toReport.write (*os);
		if (! xml || ! where)
      *os << ']';
	}

	if ((bool)where) { // list graph ids for pattern
	  if (xml) // close tag
	    *os << "</what>\n<where>";
	  else
	    *os << " {";

	  if (where == 1)
	    for (std::map<unsigned int, unsigned int>::const_iterator cur_iter = supp_counts.begin(); cur_iter != supp_counts.end(); ++cur_iter) {
	      assert( originalInstanceLabels.size() == 0 
		      || cur_iter->first < originalInstanceLabels.size() );

	      if (cur_iter != supp_counts.begin())
          *os << ' ';
	      *os << (originalInstanceLabels.size() == 0 ? 
      		      cur_iter->first :
      		      originalInstanceLabels.at(cur_iter->first));
	    }
	  else { // must be 2
	    for (std::map<unsigned int, unsigned int>::const_iterator cur_iter = supp_counts.begin(); cur_iter != supp_counts.end(); ++cur_iter) {
	      assert( originalInstanceLabels.size() == 0 
		      || cur_iter->first < originalInstanceLabels.size() );

	      if (cur_iter != supp_counts.begin())
          *os << ' ';
	      *os <<  (originalInstanceLabels.size() == 0 ? 
                cur_iter->first :
                originalInstanceLabels.at(cur_iter->first)) << 
            ':' << cur_iter->second;
	    }
	  }
	  if (xml)
	    *os << "</where>\n</pattern>";
	  else
	    *os << '}';
	}

	*os << '\n';
	++ID;
}

/**
  * Get the maximally possible improvement achieved by the extensions stored in 
  * correspondence class cor.
  */
unsigned int getPartialImprovment(const Correspondence * cor) {
  // improvement = max { 0,     A1 * (B1 - B0),     B1 * (A1 - B0) } (for all classes; 1-vs.-rest)
  unsigned int tempImprovement = 0, s = (cor->counts).size(), subImprovement;
  int a_bdiff, b_adiff;
  unsigned int total_counts = 0, total_ext = 0;
  for (unsigned int i = 0; i < s; i++) { 
    total_counts += (cor->counts)[i];
    total_ext += (cor->ext)[i];
  }
  for (unsigned int i = 0; i < s; i++) { // class A  ---   B is the rest of the dataset
    subImprovement = 0; // no improvement
    // A1 * (B1 - B0) = A1 * (B1 - (B - B1)) = A1 * (2 B1 - B)
    a_bdiff = (cor->ext)[i] * (2*(total_ext - (cor->ext)[i]) - (total_counts - (cor->counts)[i]));
    if (a_bdiff > 0)
      subImprovement = static_cast<unsigned int>( a_bdiff );
    // B1 * (A1 - A0) = B1 * (A1 - (A - A1)) = B1 * (2 A1 - A)
    b_adiff = (total_ext - (cor->ext)[i]) * (2*(cor->ext)[i] - (cor->counts)[i]);
    if (b_adiff > static_cast<int>(subImprovement))
      subImprovement = static_cast<unsigned int>( b_adiff );
    tempImprovement += subImprovement;
  }
  return tempImprovement;
}

/**
  * Multi-class correspondences are: 
  * sum_{class a} #correspondences_{"a" vs. "not a"}
  */
unsigned int oneAgainstRestCorresondences(std::vector<unsigned int> const & classes) {
  unsigned int corrs = 0, numMatches = 0;
  for (std::vector<unsigned int>::const_iterator cIt = classes.begin(); cIt != classes.end(); cIt++)
    numMatches += *cIt; // get total number of matches
  for (std::vector<unsigned int>::const_iterator cIt = classes.begin(); cIt != classes.end(); cIt++) {
    corrs += *cIt * (numMatches - *cIt);
  }
  return corrs;
}

/*
 * Returns true if any child graph of the current subgraph represented in
 *   "projected" (or, if "use_conserved", in "current_best_support_counts")
 *   can improve the current CORK-value by more than the current
 *   maximal CORK value - if "get_cork_value" != NULL, it gets assigned the
 *   new CORK value of the current subgraph.
 */
bool gSpan::getAskCORK(Projected const * projected,
    unsigned int * get_cork_value,
    bool use_conserved) {

  /* identify the effects of all matching graphs on the correspondence
       * equivalence classes */
  unsigned int oid = 0xffffffff;
  Correspondence * cor;
  if (use_conserved) {
    for (std::map<unsigned int, unsigned int>::const_iterator c_iter = current_best_support_counts.begin(); c_iter != current_best_support_counts.end(); c_iter++) {
      assert(equ_classes.size() > c_iter->first && class_labels.size() > c_iter->first && equ_classes.size() > c_iter->first && possible_changes.size() > c_iter->first && correspondence_classes.size() > equ_classes[c_iter->first]);
      cor = &(correspondence_classes[equ_classes[c_iter->first]]);
      (cor->ext).resize((cor->counts).size()); // ensure ext is initialized
      (cor->ext)[class_labels[c_iter->first]]++;
      possible_changes[c_iter->first] = true; // prepare extend() step
    }
  } else {
    for (Projected::const_iterator cur = projected->begin(); cur != projected->end(); ++cur) {
      if (oid != cur->id) {
      	assert(equ_classes.size() > cur->id && class_labels.size() > cur->id && equ_classes.size() > cur->id && possible_changes.size() > cur->id && correspondence_classes.size() > equ_classes[cur->id]);
        cor = &(correspondence_classes[equ_classes[cur->id]]);
        (cor->ext).resize((cor->counts).size()); // ensure ext is initialized
        (cor->ext)[class_labels[cur->id]]++;
      	oid = cur->id;
      }
    }
  }

  if (get_cork_value == NULL) // only re-freshing extension information - already passed CORK test
    return true;

  /* calculate current CORK value and possible further gain */

  *get_cork_value = 0;
  unsigned int maxImprovement = 0, tempImprovement = 0;
  for (std::vector<Correspondence>::iterator cor_iter = correspondence_classes.begin(); cor_iter != correspondence_classes.end(); cor_iter++) {
    //  compute CORK
    if ((cor_iter->ext).size() == 0) {
      (cor_iter->ext).resize((cor_iter->counts).size());
    }
    assert((cor_iter->ext).size() == (cor_iter->counts).size());
    
    std::vector<unsigned int> misses = std::vector<unsigned int>(cor_iter->counts);
    std::vector<unsigned int>::const_iterator hIt = (cor_iter->ext).begin();
    for (std::vector<unsigned int>::iterator mIt = misses.begin(); mIt != misses.end(); mIt++) {
      *mIt -= *hIt;
      hIt++;
    }
    *get_cork_value += oneAgainstRestCorresondences(cor_iter->ext);
    *get_cork_value += oneAgainstRestCorresondences(misses);
    
    tempImprovement = getPartialImprovment(&(*cor_iter));

    maxImprovement += tempImprovement;
  }

  /* test CORK pruning threshold */
  assert(maxImprovement <= *get_cork_value);
  if (*get_cork_value - maxImprovement < correspondences)
    return true;
  return false;
}

void gSpan::resetCORK_Extensions() {
    for (std::vector<Correspondence>::iterator cor_iter = correspondence_classes.begin(); cor_iter != correspondence_classes.end(); cor_iter++) {
      (cor_iter->ext).assign((cor_iter->counts).size(),0);
    }
}

void gSpan::extendCORK() {
  std::vector<unsigned int> correspondence_children(correspondence_classes.size());

  unsigned int numCs = correspondence_classes.size();
  for (unsigned int i = 0; i < numCs; i++) {
    assert(i < correspondence_classes.size());
    
    unsigned int num0 = 0, extNot0 = 0, numEqMax = 0;
    for(unsigned int j = 0; j < correspondence_classes[i].counts.size(); j++) {
      if (correspondence_classes[i].ext[j] == correspondence_classes[i].counts[j])
        numEqMax++;
      if (correspondence_classes[i].ext[j] != 0)
        extNot0++;
      if (correspondence_classes[i].counts[j] == 0)
        num0++;
    }
    
    if (extNot0 > 0) {
      if (numEqMax == correspondence_classes[i].counts.size()) {
        correspondence_classes[i].ext.assign(correspondence_classes[i].counts.size(), 0);
        continue; // not a really new equivalence class
      }

      /* possibility of avoiding meaningless correspondence classes: */
      if (delete_resolved)
      	if (num0 >= correspondence_classes[i].counts.size()-1) {
            // split up only slows us down; however this means that we will not know the correct number of patterns generated by the selected subgraphs
      	  correspondence_classes[i].ext.assign(correspondence_classes[i].counts.size(), 0);
      	  continue;
      	}
      /*****/

      correspondence_children[i] = correspondence_classes.size();

      // build and fill new correspondence
      correspondence_classes.resize(correspondence_classes.size()+1);
      correspondence_classes[correspondence_classes.size()-1].counts = correspondence_classes[i].ext;
      correspondence_classes[correspondence_classes.size()-1].father = i;
      
      // reduce "old" correspondence
      for (unsigned int j = 0; j < correspondence_classes[i].counts.size(); j++)
        correspondence_classes[i].counts[j] -= correspondence_classes[i].ext[j];
      correspondence_classes[i].ext.assign(correspondence_classes[i].counts.size(), 0);
    }
  }

  // assign new correspondence equivalence classes
  for (unsigned int i = 0; i < equ_classes.size(); i++) {
    if (possible_changes[i]) {
      if (correspondence_children[equ_classes[i]] != 0)
        equ_classes[i] = correspondence_children[equ_classes[i]];
      possible_changes[i] = false;
    }
  }
  
  // delete all graphs which are not further participating in feature selection
  if (delete_resolved) {
    std::vector<unsigned int>::iterator clIter = class_labels.begin();
    std::vector<unsigned int>::iterator ecIter = equ_classes.begin(), oilIter = originalInstanceLabels.begin(), oirIter = originalInstanceRanks.begin();
    std::vector < Graph >::iterator tIter = TRANS.begin();
    while (ecIter != equ_classes.end()) {
      unsigned int num0 = 0;
      for(unsigned int j = 0; j < correspondence_classes[*ecIter].counts.size(); j++) {
        if (correspondence_classes[*ecIter].counts[j] == 0)
          num0++;
      }
      if (num0 >= correspondence_classes[*ecIter].counts.size()-1) { // resolved
      	ecIter = equ_classes.erase(ecIter);
      	clIter = class_labels.erase(clIter);
      	tIter = TRANS.erase(tIter);
      	if (oilIter != originalInstanceLabels.end())
      	  oilIter = originalInstanceLabels.erase(oilIter);
      	if (oirIter != originalInstanceRanks.end())
      	  oirIter = originalInstanceRanks.erase(oirIter);
      } else {
      	ecIter++;
      	clIter++;
      	tIter++;
      	if (oilIter != originalInstanceLabels.end())
      	  oilIter++;
        if (oirIter != originalInstanceRanks.end())
          oirIter++;
      }
    }
    possible_changes.resize(TRANS.size());
  }

}


/* Recursive subgraph mining function (similar to subprocedure 1
 * Subgraph_Mining in [Yan2002]).
 */
void gSpan::project (Projected &projected) {
  if (verbose)
    std::cout<<"project"<<std::endl;

  tested_subgraphs++;

	/* Check if the pattern is frequent enough.
	 */
	unsigned int sup = support (projected);
	if (sup < minsup)
		return;

	frequent_subgraphs++;

	if (verbose)
 	  std::cout<<"frequent ("<<sup<<")"<<std::endl;


	/* The minimal DFS code check is more expensive than the support check,
	 *  hence it is done now, after checking the support.
	 */
	if (is_min () == false) {
	  return;
	}

	if (verbose)
	  std::cout<<"minimal"<<std::endl;

	minimal_subgraphs++;
	
	if (fs) { // exploit feature selection pruning

	  assert(equ_classes.size() == class_labels.size() && class_labels.size() == possible_changes.size());

	  if (1) { // enable further selection options

	    unsigned int corrs = 0;
	    bool testCORK = getAskCORK(&projected, &corrs);

	    if (verbose) {
	      std::cout<<corrs<<" correspondences"<<std::endl;
	    }

	    if ( corrs < correspondences ) { // found better subgraph
    		correspondences = corrs;
    		if (verbose)
    		  std::cout<<"new winner"<<std::endl;
    		
    		winner_subgraphs++;
    		
    		// TODO: more efficiency via explicit vector copy calls?
    		current_best_support_counts = support_counts(projected);
    		current_best_subgraph = DFS_CODE;
    		// might be avoided ...
	    }

	    // TODO: make more efficient - using projected
	    resetCORK_Extensions(); // reset to zero-extensions
		
	    if (verbose)
	      std::cout<<"reset"<<std::endl;

	    if (!testCORK) { // failed CORK-test
	      if (verbose)
      		std::cout<<"failed test"<<std::endl;
        prunedFS_subgraphs++;
	      return; // do not extend
	    }

	    if (verbose)
	      std::cout<<"passed test"<<std::endl;

	  }
	} else // output is delayed or omitted
	  report (projected, sup); // Output the frequent substructure



	/* In case we have a valid upper bound and our graph already exceeds it,
	 * return.  Note: we do not check for equality as the DFS exploration may
	 * still add edges within an existing subgraph, without increasing the
	 * number of nodes.
	 */
	if (maxpat_max > maxpat_min && DFS_CODE.nodeCount () > maxpat_max)
		return;



	/* We just outputted a frequent subgraph.  As it is frequent enough, so
	 * might be its (n+1)-extension-graphs, hence we enumerate them all.
	 */
	const RMPath &rmpath = DFS_CODE.buildRMPath ();
	int minlabel = DFS_CODE[0].fromlabel;
	int maxtoc = DFS_CODE[rmpath[0]].to;

	Projected_map3 new_fwd_root;
	Projected_map2 new_bck_root;
	EdgeList edges;

	/* Enumerate all possible one edge extensions of the current substructure.
	 */
	for (unsigned int n = 0; n < projected.size(); ++n) { // for all fitting graphs

		unsigned int id = projected[n].id;
		PDFS *cur = &projected[n];
		History history (TRANS[id], cur);

		// XXX: do we have to change something here for directed edges?

		// backward
		for (int i = (int)rmpath.size()-1; i >= 1; --i) {
			Edge *e = get_backward (TRANS[id], history[rmpath[i]], history[rmpath[0]], history);
			if (e)
				new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push (id, e, cur);
		}

		// pure forward
		// FIXME: here we pass a too large e->to (== history[rmpath[0]]->to
		// into get_forward_pure, such that the assertion fails.
		//
		// The problem is:
		// history[rmpath[0]]->to > TRANS[id].size()
		if (get_forward_pure (TRANS[id], history[rmpath[0]], minlabel, history, edges))
			for (EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
				new_fwd_root[maxtoc][(*it)->elabel][TRANS[id][(*it)->to].label].push (id, *it, cur);

		// backtracked forward
		for (int i = 0; i < (int)rmpath.size(); ++i)
			if (get_forward_rmpath (TRANS[id], history[rmpath[i]], minlabel, history, edges))
				for (EdgeList::iterator it = edges.begin(); it != edges.end();  ++it)
					new_fwd_root[DFS_CODE[rmpath[i]].from][(*it)->elabel][TRANS[id][(*it)->to].label].push (id, *it, cur);
	}

	/* Test all extended substructures.
	 */
	// backward
	for (Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
		for (Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) { // mis-used parameter names ...
			DFS_CODE.push (maxtoc, to->first, -1, elabel->first, -1);
			project (elabel->second);
			DFS_CODE.pop();
		}
	}

	// forward
	for (Projected_riterator3 from = new_fwd_root.rbegin() ;
      from != new_fwd_root.rend() ; ++from)	{
		for (Projected_iterator2 elabel = from->second.begin() ;
        elabel != from->second.end() ; ++elabel) {
			for (Projected_iterator1 tolabel = elabel->second.begin();
					tolabel != elabel->second.end(); ++tolabel)	{
				DFS_CODE.push (from->first, maxtoc+1, -1, elabel->first, tolabel->first);
				project (tolabel->second);
				DFS_CODE.pop ();
			}
		}
	}

	return;
}


std::vector<unsigned int> gSpan::assign_class_labels(std::string class_label_file, std::set<unsigned int> const * graphFilter) {
  if (TRANS.size() == 0) {
    std::cerr<< "error: no graphs read in before assigning graph class labels (gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
    exit(1);
  }
  std::vector<unsigned int> class_counts;
  if (class_label_file=="") {
    // class labels must already have been set;
    // only assert dimensions and update class counts
    if (class_labels.size() != TRANS.size()) {
      std::cerr<< "error: # mapped class labels ("<<class_labels.size()<<") must be equal zu #graphs ("<<TRANS.size()<<") (gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
      exit(1);
    }
    std::set<unsigned int> disjunctClasses;
    unsigned int max_label = 0;
    for (unsigned int i = 0; i < class_labels.size(); i++) {
      if (class_counts.size() <= class_labels[i])
        class_counts.resize(class_labels[i]+1);
      class_counts[class_labels[i]]++;
      disjunctClasses.insert(class_labels[i]);
      if (max_label < class_labels[i])
        max_label = class_labels[i];
    }
    number_of_classes = disjunctClasses.size();
    if (max_label != number_of_classes-1) {
      std::cerr<< "error: require a class labelling from 0 to (num_classes-1) " << max_label <<"!=" << (number_of_classes-1) <<"(gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
      exit(1);
    }
    return class_counts;
  }

  class_labels.resize(TRANS.size());

  std::ifstream classLabelFile(class_label_file.c_str());
  if (!classLabelFile) {
    std::cerr<< "error: cannot open file containing graph labels \""<<class_label_file<<"\" (gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
    exit(1);
  }
  std::string str, tempStr;

  getline(classLabelFile, str);
  std::istringstream iss(str);
  unsigned int graph_id = 0, /* accepted */ graph_index = 0 /* encountered */;
  if (!(iss>>tempStr)) { // take first word of the line
    tempStr=str; // one entry only
    if (tempStr=="") {
      std::cerr << "error: graph id mapping file \"" << class_label_file << "\" does not contain" << std::endl;
      std::cerr << "       a class label for graph " << graph_id << " (gSpan::assign_class_labels:" <<  __LINE__ << ")" << std::endl;
      exit(1);
    }
  }
  if (tempStr[0] == '-') {
    std::cerr << "error: graph id mapping file \""<<class_label_file<<"\" has a negative"<<std::endl;
    std::cerr << "       class label \""<<tempStr<<"\" for graph "<<graph_index<<" (gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
    exit(1);
  }
  unsigned int classLabel = strtoul(tempStr.c_str(),NULL,10);
  if (graphFilter == NULL || graphFilter->find(graph_index) != graphFilter->end()) {
    class_labels[graph_id] = classLabel;
    class_counts.resize(classLabel+1);
    class_counts[classLabel]++;
    graph_id++;
  }
  graph_index++;
  
  std::set<unsigned int> disjunctClasses;
  unsigned int max_label = 0;

  while(getline(classLabelFile, str)) {
    if (graphFilter != NULL && graphFilter->find(graph_index) == graphFilter->end()) {
      graph_index++; // skip this graph
      continue;
    }
    iss.str(str);
    if (!(iss>>tempStr)) { // take first word of the line
      tempStr=str; // one entry only
      if (tempStr=="") {
      	std::cerr<< "error: graph id mapping file \""<<class_label_file<<"\" does not contain a class label for graph "<<graph_index<<" (gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
      	std::cerr<< "       '"<<str<<"' to '"<<tempStr<<"' from '"<<iss.str()<<"'"<<std::endl;
      	exit(1);
      }
    }
    if (tempStr[0] == '-') {
      std::cerr << "error: graph id mapping file \""<<class_label_file<<"\" has a negative"<<std::endl;
      std::cerr << "       class label \""<<tempStr<<"\" for graph "<<graph_index<<" (gSpan::assign_class_labels:"<< __LINE__<<")"<<std::endl;
      exit(1);
    }
    classLabel = strtoul(tempStr.c_str(), NULL, 10);
    class_labels[graph_id++] = classLabel;
    if (class_counts.size() <= classLabel)
      class_counts.resize(classLabel+1);
    class_counts[classLabel]++;
    disjunctClasses.insert(classLabel);
    if (max_label < classLabel)
      max_label = classLabel;
    graph_index++;
  }
  classLabelFile.close();

  number_of_classes = disjunctClasses.size();
  if (max_label != disjunctClasses.size()-1) {
    // must re-label the class labels (empty class slots cost time)
    unsigned int new_labels[max_label+1];
    unsigned int class_index = 0;
    for (std::set<unsigned int>::const_iterator clIt = disjunctClasses.begin(); clIt != disjunctClasses.end(); clIt++) {
      class_counts[class_index] = class_counts[*clIt];
      new_labels[*clIt] = class_index++;
    }
    for (std::vector<unsigned int>::iterator clIt = class_labels.begin(); clIt != class_labels.end(); clIt++) {
      class_index = new_labels[*clIt];
      *clIt = class_index;
    }
    class_counts.resize(disjunctClasses.size());
  }
  return class_counts;
}

std::map<unsigned int,unsigned int> gSpan::translateSupportCounts(std::map<unsigned int,unsigned int> const & _instances2classLabels) {
  if (originalInstanceRanks.size() == 0) {
    unsigned int i = 0;
    originalInstanceRanks.resize(originalInstanceLabels.size());
    std::map<unsigned int,unsigned int> lab2rank;
    for (std::map<unsigned int,unsigned int>::const_iterator i2cIt = instances2classLabels.begin(); i2cIt != instances2classLabels.end(); i2cIt++)
      lab2rank[i2cIt->first] = i++;
    for (i = 0; i < originalInstanceLabels.size(); i++)
      originalInstanceRanks.at(i) = lab2rank[originalInstanceLabels[i]];
  }
  std::map<unsigned int,unsigned int> supports_translated;
  for (std::map<unsigned int,unsigned int>::const_iterator i2cIt = _instances2classLabels.begin(); i2cIt != _instances2classLabels.end(); i2cIt++)
    supports_translated[originalInstanceRanks.at(i2cIt->first)] = i2cIt->second;

  return supports_translated;
}

void gSpan::setInfoStream(std::ostream &infoS) {
  infoStream = &infoS;
}

void gSpan::run (std::istream &is, std::ostream &_os,
		 unsigned int _minsup,
		 unsigned int _maxpat_min, unsigned int _maxpat_max,
		 bool _enc,
		 unsigned int _where,
     bool _xml,
		 bool _directed, std::set<unsigned int> const * graphFilter,
		 bool _verbose) {

  os = &_os;
	ID = 0;
	minsup = _minsup;
	maxpat_min = _maxpat_min;
	maxpat_max = _maxpat_max;
	enc = _enc;
	xml = _xml;
	where = _where;
	directed = _directed;
	verbose = _verbose;

	read (is, graphFilter);
	run_intern ();

}

std::vector<DFSCode> gSpan::run_gSpan_CORK (std::istream &is, std::ostream &_os,
			  unsigned int _minsup,
			  unsigned int _maxpat_min, unsigned int _maxpat_max,
			  bool _enc, unsigned int _where,
			  bool _xml, bool _directed,
        std::set<unsigned int> const * graphFilter,
			  std::string _fs_option, std::string class_label_file,
			  std::vector<std::map<unsigned int, unsigned int> > * supportMap,
			  bool _verbose) {

  os = &_os;

  fs = false;
  ID = 0;
  minsup = _minsup;
  maxpat_min = _maxpat_min;
  maxpat_max = _maxpat_max;
  enc = _enc;
  xml = _xml;
  where = _where;
  directed = _directed;
  verbose = _verbose;
  delete_resolved = false;
  
  time_t time_now = time(NULL);

  read (is, graphFilter); // neither is this

  double minsupRatio = static_cast<double>(minsup) / TRANS.size();
  
  if (verbose)
    std::cout << "read graphs; graph size: "<<TRANS.size()<<", time: "<< difftime(time_now, time(NULL)) <<std::endl;

  if (_fs_option != "") { // perform nested feature selection

    fs = true;
    fs_option = _fs_option;
    unsigned int selection_threshold = 0; // # subgraphs to select
    unsigned int correspondence_threshold = 0; // # correspondences allowed
    if ( (fs_option.size() > 4 && fs_option.substr(0,4) == "CORK") ||
         (fs_option.size() > 5 && fs_option.substr(0,5) == "FCORK") ) {
      if (verbose)
        std::cout<<"reading in class labels"<<std::endl;
      if (fs_option.size() > 5 && fs_option.substr(0,5) == "FCORK")
        delete_resolved = true;
      selection_threshold = strtoul(fs_option.substr((delete_resolved ? 5 : 4)).c_str(), NULL, 10);
      if (fs_option.size() > 4 && fs_option.find('C', 4) != std::string::npos && fs_option.find('C', 4) != fs_option.size())
        correspondence_threshold = strtoul(fs_option.substr(fs_option.find('C', 4)+1).c_str(), NULL, 10);
    } else if (fs_option.size() < 4 || 
               ( (fs_option.size() == 4 && fs_option.substr(0,4) != "CORK") || 
                 (fs_option.size() == 5 && fs_option.substr(0,5) != "FCORK") ) ) {
      std::cerr << "error: do not know option \""<<fs_option<<"\" (gSpan::run_gSpan_CORK:"<< __LINE__<<")"<<std::endl;
      exit(1);
    }
    if (selection_threshold == 0) {
      selection_threshold = std::numeric_limits<unsigned int>::max();
      *infoStream << "warning: no upper limit for the maximum number of selected subgraphs given;"<<std::endl;
      *infoStream << "         using " << selection_threshold << " (gSpan::run_gSpan_CORK:" << __LINE__ << ")" << std::endl;
    }
    
    if (verbose)
      std::cout<<"starting with nested feature selection"<<std::endl;

    std::vector<unsigned int> class_counts = assign_class_labels(class_label_file, graphFilter);

    if (verbose)
      std::cout<<"labels assigned"<<std::endl;

    equ_classes = std::vector<unsigned int>(TRANS.size());
    possible_changes = std::vector<bool>(TRANS.size());
    correspondence_classes.resize(1); // 1st correspondence
    correspondence_classes[0].father = -1; // no original corresponding
    correspondence_classes[0].counts = class_counts;
    correspondences = oneAgainstRestCorresondences(class_counts);

    selectedSubgraphs.clear();

    *infoStream<<"\tcorrespondences\tcorrespondence_classes\tnum_unresolved\tsize_unresolved\tmax_unresolved\titeration\ttested\tfrequent\tminimal\twinners\tpruned\ttime_[s]"<<std::endl;

    time_now = time(NULL);
    
    // perform greedy forward selection: one feature == one pruned gSpan DFS code tree traversal
    for (unsigned int i = 0; i < selection_threshold; i++) {

      tested_subgraphs = 0;
      frequent_subgraphs = 0;
      minimal_subgraphs = 0;
      winner_subgraphs = 0;
      prunedFS_subgraphs = 0;

      current_best_subgraph = DFSCode(); // initialize with the empty subgraph

      if (verbose)
        std::cout<<"iteration "<<i<<std::endl;

      // run gSpan
      run_intern ();

      if (current_best_subgraph.size() == 0) { // no further best graph found
      	*infoStream<<"note: terminating early: "<<i<<" instead of "<<selection_threshold<<" frequent subgraphs selected"<<std::endl;
      	break;
      }

      // report selected subgraph
      ID = i;
      // if wanted: subgraphs can be reported online here
      report(current_best_subgraph, current_best_support_counts);
      // determine the number of unresolved correspondences
      double numUnres = 0, corrsUnres = 0;
      unsigned int unresMax = 0;
      unsigned int numNotNull;
      for (std::vector<Correspondence>::iterator cor_iter = correspondence_classes.begin(); cor_iter != correspondence_classes.end(); cor_iter++) {
      	unsigned int currentCorrSize = 0;
        numNotNull = 0;
        for (std::vector<unsigned int>::const_iterator cIt = cor_iter->counts.begin();
              cIt != cor_iter->counts.end(); cIt++) {
              currentCorrSize += *cIt;
              if (*cIt != 0)
                numNotNull++;
        }
      	if (numNotNull > 1) { // at least two more class conflicts remain to be resolved
      	  corrsUnres += currentCorrSize;
      	  numUnres++;
      	  if (unresMax < currentCorrSize)
      	    unresMax = currentCorrSize;
      	}
      }
      corrsUnres /= numUnres;
      *infoStream<<"\t"<<correspondences<<"\t"<<correspondence_classes.size()<<"\t"<<numUnres<<"\t"<<corrsUnres<<"\t"<<unresMax<<"\t"<< i<<"\t"<<tested_subgraphs<<"\t"<<frequent_subgraphs<<"\t"<<minimal_subgraphs<<"\t"<<winner_subgraphs<<"\t"<<prunedFS_subgraphs<<"\t"<<difftime(time(NULL), time_now)<<std::endl;

      // collect output graphs
      selectedSubgraphs.push_back(current_best_subgraph);
      if (supportMap != NULL) // and the traceback
        supportMap->push_back(translateSupportCounts(current_best_support_counts));
        
      time_now = time(NULL);

      // reset variables
      DFS_CODE.clear();
      DFS_CODE_IS_MIN.clear();
      GRAPH_IS_MIN.clear();
      
      // ensure the extension of the correspondence classes by the currently 
      // selected subgraph
      getAskCORK(NULL, NULL, true); // call for extension only
      extendCORK(); // extend

      if (delete_resolved) {
        // Adapting minsup to the number of remaining graphs is not helpful for 
        // small graph collections and ought to be used with care!
      	minsup = static_cast<unsigned int>(floor(minsupRatio*TRANS.size()+.5));
      	if (minsup == 0) // should not happen
      	  minsup = 1; 
      }
      
      if (correspondence_threshold > correspondences)
        break;
    }

    // collect final reporting output
    double numUnres = 0, corrsUnres = 0;
    unsigned int unresMax = 0, numNotNull;
    for (std::vector<Correspondence>::iterator cor_iter = correspondence_classes.begin(); cor_iter != correspondence_classes.end(); cor_iter++) {
      unsigned int currentCorrSize = 0;
      numNotNull = 0;
      for (std::vector<unsigned int>::const_iterator cIt = cor_iter->counts.begin();
            cIt != cor_iter->counts.end(); cIt++) {
            currentCorrSize += *cIt;
            if (*cIt != 0)
              numNotNull++;
      }
      if (numNotNull > 1) {
        corrsUnres += currentCorrSize;
        numUnres++;
        if (unresMax < currentCorrSize)
          unresMax = currentCorrSize;
      }
    }
    corrsUnres /= numUnres;
    *infoStream<<"\t"<<correspondences<<"\t"<<correspondence_classes.size()<<"\t"<<numUnres<<"\t"<<corrsUnres<<"\t"<<unresMax<<"\tlast\t"<<tested_subgraphs<<"\t"<<frequent_subgraphs<<"\t"<<minimal_subgraphs<<"\t"<<winner_subgraphs<<"\t"<<prunedFS_subgraphs<<"\t"<<difftime(time(NULL), time_now)<<std::endl;

  } else { // no feature selection

    if (verbose)
      *infoStream << "running gSpan without feature selection" << std::endl;

    run_intern ();
  }
  return selectedSubgraphs;
}

std::vector<DFSCode> gSpan::run_gSpan_CORK (std::istream &is, std::ostream &_os,
			  unsigned int _minsup,
			  unsigned int _maxpat_min, unsigned int _maxpat_max,
			  bool _enc, unsigned int _where,
			  bool _xml, bool _directed,
        std::set<unsigned int> const * graphFilter,
			  std::string _fs_option,
			  std::map<unsigned int,unsigned int> const * _instances2classLabels,
			  std::vector<std::map<unsigned int, unsigned int> > * supportMap,
			  bool _verbose) {
  if (_instances2classLabels != NULL)
    instances2classLabels = std::map<unsigned int,unsigned int>(*_instances2classLabels);
  return run_gSpan_CORK (is, _os, _minsup, _maxpat_min, _maxpat_max, _enc, _where, _xml, _directed, graphFilter, _fs_option, "", supportMap,_verbose);
}


void gSpan::run_intern (void) {
	/* In case 1 node subgraphs should also be mined for, do this as
	 * preprocessing step.
	 */
	if (maxpat_min <= 1) {
		/* Do single node handling, as the normal gspan DFS code based processing
		 * cannot find subgraphs of size |subg|==1.  Hence, we find frequent node
		 * labels explicitly.
		 */
		for (unsigned int id = 0; id < TRANS.size(); ++id) {
			for (unsigned int nid = 0 ; nid < TRANS[id].size() ; ++nid) {
				if (singleVertex[id][TRANS[id][nid].label] == 0) {
					// number of graphs it appears in
					singleVertexLabel[TRANS[id][nid].label] += 1;
				}

				singleVertex[id][TRANS[id][nid].label] += 1;
			}
		}
		/* All minimum support node labels are frequent 'subgraphs'.
		 * singleVertexLabel[nodelabel] gives the number of graphs it appears
		 * in.
		 *
		 * 1/1.5-class case: All nodelabels that do not appear at all have a
		 *    gain of zero, hence we do not need to consider them.
		 *
		 * 2-class case: Not appearing nodelabels are counted negatively.
		 */
		for (std::map<unsigned int, unsigned int>::iterator it =
			singleVertexLabel.begin () ; it != singleVertexLabel.end () ; ++it) {
			if ((*it).second < minsup)
				continue;

			unsigned int frequent_label = (*it).first;

			/* Found a frequent node label, report it.
			 */
			Graph g(directed);
			g.resize (1);
			g[0].label = frequent_label;

			/* [graph_id] = count for current substructure
			 */
			std::vector<unsigned int> counts (TRANS.size ());
			for (std::map<unsigned int, std::map<unsigned int, unsigned int> >::iterator it2 =
				singleVertex.begin () ; it2 != singleVertex.end () ; ++it2) {
				counts[(*it2).first] = (*it2).second[frequent_label];
			}

			std::map<unsigned int, unsigned int> gycounts;
			for (unsigned int n = 0 ; n < counts.size () ; ++n)
			  gycounts[n] = counts[n];

			report_single (g, gycounts);

		}
	} // END maxpatmin <= 1

	EdgeList edges;
	Projected_map3 root;

	for (unsigned int id = 0; id < TRANS.size(); ++id) {
		Graph &g = TRANS[id];

		for (unsigned int from = 0; from < g.size() ; ++from) {
			if (get_forward_root (g, g[from], edges)) {
				for (EdgeList::iterator it = edges.begin(); it != edges.end();  ++it)
					root[g[from].label][(*it)->elabel][g[(*it)->to].label].push (id, *it, 0);
			}
		}
	}

	for (Projected_iterator3 fromlabel = root.begin() ;
       fromlabel != root.end() ; ++fromlabel) {
		for (Projected_iterator2 elabel = fromlabel->second.begin() ;
         elabel != fromlabel->second.end() ; ++elabel)	{
			for (Projected_iterator1 tolabel = elabel->second.begin();
				   tolabel != elabel->second.end(); ++tolabel)	{
				/* Build the initial two-node graph.  It will be grown
				 *  recursively within the project.
				 */
				DFS_CODE.push (0, 1, fromlabel->first, elabel->first, tolabel->first);
				project (tolabel->second);
				DFS_CODE.pop ();
			}
		}
	}
}

}
