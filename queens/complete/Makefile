CC = g++
CFLAGS = -O2 -pipe -march=native -std=c++14

clean:
	mkdir clean
	mv * clean

nqueens.exe: nqueens.cpp
	g++ -O2 -march=native -std=c++14 nqueens.cpp -o nqueens.exe
    
n2queens.exe: n2queens.cpp
	g++ -O2 -march=native -std=c++14 n2queens.cpp -o n2queens.exe

n2queens2.exe: n2queens2.cpp
	g++ -O2 -march=native -std=c++14 n2queens2.cpp -o n2queens2.exe

n2queens3.exe: n2queens3.cpp
	g++ -O2 -march=native -std=c++14 n2queens3.cpp -o n2queens3.exe
	
n2queens4.exe: n2queens4.cpp
	g++ -O2 -march=native -std=c++14 n2queens4.cpp -o n2queens4.exe
	
n2queens5.exe: n2queens5.cpp
	g++ -O2 -march=native -std=c++14 n2queens5.cpp -o n2queens5.exe


n2queens_partial.exe: n2queens_partial.cpp
	$(CC) ${CFLAGS} n2queens_partial.cpp -o n2queens_partial.exe

#n2queens5.exe: n2queens5.o 5helper.o
#	$(CC) ${CFLAGS} n2queens5.o 5helper.o -o n2queens5.exe
#
#5helper.o: 5helper.cpp 5helper.h
#	$(CC) ${CFLAGS} -c 5helper.cpp
#	
#n2queens5.o: n2queens5.cpp 5helper.h
#	$(CC) ${CFLAGS} -c n2queens5.cpp
