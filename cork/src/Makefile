############################################################################
###   Copyright (C) 2004 Taku Kudo, All rights reserved.
###     This is free software with ABSOLUTELY NO WARRANTY.
###  
###   This program is free software; you can redistribute it and/or modify
###     it under the terms of the GNU General Public License as published by
###     the Free Software Foundation; either version 2 of the License, or
###     (at your option) any later version.
###    
###   This program is distributed in the hope that it will be useful,
###     but WITHOUT ANY WARRANTY; without even the implied warranty of
###     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
###     GNU General Public License for more details.
###    
###   You should have received a copy of the GNU General Public License
###     along with this program; if not, write to the Free Software
###     Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
###     02111-1307, USA
###
###     modified by Marisa Thoma (extracted the original version from matlab; 
###                                                 extended to CORK pruning)
###    release: 2010/11/09
###    version: 1.2
############################################################################

CPP = g++
BIN = ../bin/

###
### The values below should not be modified;
### However, of course, optimization options can greatly improve runtimes
###

# DEBUG and RELEASE settings
#CPPFLAGS = -DDEBUG -g -ggdb -Wall -Wno-deprecated
CPPFLAGS = -DNDEBUG -Os -g -ggdb -Wall -Wno-deprecated

OBJ = ${BIN}graph.o ${BIN}misc.o ${BIN}dfs.o ${BIN}ismin.o ${BIN}gspan.o

### Targets

all:  ${BIN}gSpanCORK

${BIN}gSpanCORK: ${OBJ} gSpanCORK.cpp gspan.h
	${CPP} ${CPPFLAGS} gSpanCORK.cpp ${OBJ} -o ${BIN}gSpanCORK

${BIN}gspan.o: gspan.cpp gspan.h
	${CPP} ${CPPFLAGS} -c gspan.cpp -o ${BIN}gspan.o
${BIN}graph.o: graph.cpp gspan.h
	${CPP} ${CPPFLAGS} -c graph.cpp -o ${BIN}graph.o
${BIN}misc.o: misc.cpp gspan.h
	${CPP} ${CPPFLAGS} -c misc.cpp -o ${BIN}misc.o
${BIN}dfs.o: dfs.cpp gspan.h
	${CPP} ${CPPFLAGS} -c dfs.cpp -o ${BIN}dfs.o
${BIN}ismin.o: ismin.cpp gspan.h
	${CPP} ${CPPFLAGS} -c ismin.cpp -o ${BIN}ismin.o


clean:
	rm -f ${BIN}*.o

