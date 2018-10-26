import sys
import os.path
import numpy as np
import subprocess

n = int(sys.argv[1])

DEBUG_is_good = False

def is_good(board,row,col):
    # col
    if np.sum(board, axis=0)[col] != 0:
        return False
    # row
    if np.sum(board, axis=1)[row] != 0:
        return False
    # diag \
    off = col - row
    if np.sum(np.diagonal(board, off)) != 0:
        return False
    # diag /
    off = off = n - col - row - 1
    if np.sum(np.diagonal(np.fliplr(board), off)) != 0:
        return False
        
    return True
    

def is_solution(board):
    if np.sum(board) == n:
        return True
    return False
    
    
def view(board):
    print(board)
    print("\n")
    
def print_board(board):
    s = [[str(j) for j in i] for i in board[0]]
    with open(str(n) + "/" + str(n) + "." + str(board[1][0]) + "." + str(board[1][1]) + "." + str(board[1][2]) + ".subset","a") as f:
        for i in s:
            f.write(" ".join(i) + "\n")
        f.write("\n")
        

def make_boards():
    boards = []
    for slope in range(2,n):
        board = np.zeros(shape=(n,n),dtype=int)
        for row in range(n):
            col = (row*slope) % n
            if is_good(board,row,col):
                board[row,col] = 1
        if not is_solution(board): continue
        for r in range(2):
            for i in range(n):
                # R
                boards.append([np.roll(np.rot90(board,r),i,0), (slope,r,"R")])
                # S
                boards.append([np.roll(np.rot90(np.fliplr(board),r),i,0), (slope,r,"S")])
            
    remove = []
    print(len(boards))
    for a in range(len(boards)):
        for b in range(a+1,len(boards)):
            if np.all(boards[a][0]==boards[b][0]):
                remove.append(b)
    boards = [board for c,board in enumerate(boards) if c not in remove]
    print(len(boards))
    for board in boards:
        print_board(board)
            
if __name__ == '__main__':
    if not os.path.isdir(str(n)):
        subprocess.run(["mkdir", str(n)])
        make_boards()
    print("Done!!")
