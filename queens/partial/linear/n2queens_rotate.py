import sys
import os
import numpy as np

n = int(sys.argv[1])
fname = sys.argv[2]
f = open(fname, "r")
rotate = False
# line = f.readline()
# if line == "rotate":
#    rotate = True

r = f.readline()
if r.strip() == "rotate":
    rotate = True

boards = f.read().split("\n\n")
boards = boards[:-1]

for board in boards:
    board = np.array([[i for i in row.split(" ")] for row in board.split("\n")])
    for r in range(n):
        print("\n".join([" ".join(row) for row in np.roll(board,r,1)]) + "\n")
        print("\n".join([" ".join(row) for row in np.fliplr(np.roll(board,r,1))]) + "\n")
        if rotate:
            print("\n".join([" ".join(row) for row in np.rot90(np.roll(board,r,1))]) + "\n")
            print("\n".join([" ".join(row) for row in np.rot90(np.fliplr(np.roll(board,r,1)))]) + "\n")

exit()


# def split_iter(string):
#    return (x.group(0) for x in re.finditer(r"[A-Za-z']+", string))
# loop through cubes
for line in f:
    cube = []
    # print(line.strip())
    # loop through row
    if line != "": cube.append([int(i) for i in line.strip().split(" ") if i])
    for line in f:
        if line == "":
            break
        # print(line.strip())
        cube.append([int(i) for i in line.strip().split(" ") if i ])
    for c in cube:
        print(" ".join(str(i) for i in c))
    cube = np.array(cube, dtype=int)
    # for r in range(n):
        # print(np.roll(cube,r))
        #print(lrflip(np.roll(cube,r)))
        #if rotate:
            #print(np.rot90(np.roll(cube,r)),1)
            #print(np.rot90(lrflip(np.roll(cube,r))),1)
        
