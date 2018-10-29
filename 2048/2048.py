import random
import os
import sys
import math
import numpy as np
from NeuralNetwork import *

# global vars, constants and setup
board = {}
row_size = 4
# random.seed(1)
HP = (16,16,4)

# set up game board
for i in range(row_size): # row
    for j in range(row_size): #column
        board[(i,j)] = 0


# display function
def display():
    for i in range(row_size):
        print('\t'.join([str(board[(i,j)]) for j in range(row_size)]))
    print()


# logic function
def logic(move, NN):
    """
    char move is the move, one of any in "asdw"
    NN is a NeuralNetwork object
    """
    # print("mov", move)
    if move == 's':
        for j in range(row_size): # columns
            row_pointer = row_size-1
            for i in reversed(range(row_size-1)): # go up the rows
                if board[(i, j)] != 0:
                    # if there is a non-empty square above, and this is a zero #check
                    if board[(row_pointer, j)] == 0:
                        board[(row_pointer, j)] = board[(i, j)]
                        board[(i, j)] = 0
                        # row_pointer -= 1 # This is the new block to focus on

                    # if there is a non-empty square above, and they are not equivalent
                    elif board[(i, j)] != board[(row_pointer, j)]:
                        # while this intuitively is not a swap, without it I would need to zero board[(i,j)]
                        # that zero would cause problems if row_pointer-1 == i
                        board[(row_pointer-1, j)], board[(i, j)] = board[(i, j)], board[(row_pointer-1, j)]
                        row_pointer -= 1 # This is the new block to focus on

                    # if there is a non-empty square above, and they are the same
                    elif board[(i, j)] == board[(row_pointer, j)]:
                        board[(row_pointer, j)] += board[(i, j)]
                        board[(i, j)] = 0
                        NN.score += board[(row_pointer, j)] + math.log(board[(row_pointer, j)], 2)
    elif move == 'w':
        for j in range(row_size): # columns
            row_pointer = 0
            for i in range(1, row_size): # go down the rows
                if board[(i, j)] != 0:
                    # if there is a non-empty square above, and this is a zero
                    if board[(row_pointer, j)] == 0:
                        board[(row_pointer, j)] = board[(i, j)]
                        board[(i, j)] = 0

                    # if there is a non-empty square above, and they are not equivalent
                    elif board[(i, j)] != board[(row_pointer, j)]:
                        board[(row_pointer+1, j)], board[(i, j)] = board[(i, j)], board[(row_pointer+1, j)]
                        row_pointer += 1 # This is the new block to focus on

                    # if there is a non-empty square above, and they are the same
                    elif board[(i, j)] == board[(row_pointer, j)]:
                        board[(row_pointer, j)] += board[(i, j)]
                        board[(i, j)] = 0
                        NN.score += board[(row_pointer, j)] + math.log(board[(row_pointer, j)], 2)
    elif move == 'a':
        for i in range(row_size): # rows
            column_pointer = 0
            for j in range(1, row_size): # go right through the columns
                if board[(i, j)] != 0:
                    # if there is a non-empty square above, and this is a zero
                    if board[(i, column_pointer)] == 0:
                        board[(i, column_pointer)] = board[(i, j)]
                        board[(i, j)] = 0

                    # if there is a non-empty square above, and they are not equivalent
                    elif board[(i, j)] != board[(i, column_pointer)]:
                        board[(i, column_pointer+1)], board[(i, j)] = board[(i, j)], board[(i, column_pointer+1)]
                        column_pointer += 1 # This is the new block to focus on

                    # if there is a non-empty square above, and they are the same
                    elif board[(i, j)] == board[(i, column_pointer)]:
                        board[(i, column_pointer)] += board[(i, j)]
                        board[(i, j)] = 0
                        NN.score += board[(i, column_pointer)] + math.log(board[(i, column_pointer)], 2)
    elif move == 'd':
        for i in range(row_size): # rows
            column_pointer = row_size-1
            for j in reversed(range(row_size-1)): # go left through the columns
                if board[(i, j)] != 0:
                    # if there is a non-empty square above, and this is a zero
                    if board[(i, column_pointer)] == 0:
                        board[(i, column_pointer)] = board[(i, j)]
                        board[(i, j)] = 0

                    # if there is a non-empty square above, and they are not equivalent
                    elif board[(i, j)] != board[(i, column_pointer)]:
                        board[(i, column_pointer-1)], board[(i, j)] = board[(i, j)], board[(i, column_pointer-1)]
                        column_pointer -= 1 # This is the new block to focus on

                    # if there is a non-empty square above, and they are the same
                    elif board[(i, j)] == board[(i, column_pointer)]:
                        board[(i, column_pointer)] += board[(i, j)]
                        board[(i, j)] = 0
                        NN.score += board[(i, column_pointer)] + math.log(board[(i, column_pointer)], 2)

    else:
        print("something is wrong")


# checks to see whether there are any valid moves in a full board with no 0's
def is_game_over():
    # check the top-left square
    for i in range(row_size-1):
        for j in range(row_size-1):
            if board[(i,j)] in [board[(i+1,j)], board[(i,j+1)]]: # check the one below and to the right
                return False
    # Check the right-most column
    for j in range(row_size-1):
        if board[(row_size-1,j)] == board[(row_size-1,j+1)]:
            return False
    # Check the bottom row
    for i in range(row_size-1):
        if board[(i,row_size-1)] == board[(i+1,row_size-1)]:
            return False
    # There is no way to combine, game over
    return True


# NN controls
NNs = []
for i in range(20):
    NNs.append(NeuralNetwork(HP))

# for NN in NNs:
    # print(NN.synapses)

for step in range(100):
    for NN in NNs:
        # each NN gets 5 attempts
        for trial in range(5):
            # print("new trial")

            # set up game board
            for i in range(row_size): # row
                for j in range(row_size): #column
                    board[(i,j)] = 0

            previous_board = []
            quit = False
            # game loop
            while not quit:
                # set a new empty tile to a 2
                while True:
                    i = random.randint(0,row_size-1)
                    j = random.randint(0,row_size-1)
                    # print(i,j,board[(i,j)])
                    if board[(i,j)] != 0: continue
                    else: board[(i,j)] = 2 ; break


                # View
                # display()


                # normalize data and make a guess with nn
                state = np.array([board[(i,j)] for j in range(row_size) for i in range(row_size)])
                state[state==0] = 1
                state = np.log2(state)
                state = state / np.max(state)
                # print('\n'.join(['\t'.join([str(state[j*row_size+i]) for j in range(row_size)])for i in range(row_size)]))
                move = NN.feed(state)


                # move
                previous_board = list(board.values())
                while previous_board == list(board.values()):
                    if len(move[move == 0]) == 4:
                        if is_game_over():
                            # print("Game Over")
                            quit = True
                            break
                    logic("asdw"[move.argmax()], NN)
                    move[move.argmax()] = 0
                
                    
            # display()


    # for NN in NNs:
        # print(NN.score)

    # print("\nsorted")
    NNs = sorted(NNs, key = lambda NN : -NN.score)
    # for NN in NNs:
        # print(NN.score)

    print("best: " + str(NNs[0].score) + " from " + str(NNs[0].name))
    print("mean: " + str(np.mean([NN.score for NN in NNs])) + '\n')

    # print("\nsliced")
    # for NN in NNs[:4]:
        # print(NN.score)

    # get the four best NNs
    cull = len(NNs)//5
    NNs = NNs[:cull]
    for n in range(cull):
        NNs[n].score = 0
        # each best in show gets three children
        NNs.append(NNs[n].next_gen())
        NNs.append(NNs[n].next_gen())
        NNs.append(NNs[n].next_gen())
        # new random NN every gen
        NNs.append(NeuralNetwork(HP))

    # print("\nnew set")
    # for NN in NNs:
        # print(NN.score)

# for NN in NNs:
    # print(NN.synapses)

sorted(NNs, key = lambda NN : -NN.score)[0].save()
display()


"""
    # Input
    good_input = False
    while not good_input:
        move = input()
        if move in 'asdw': 
            logic(move)
            good_input = True
        elif move == 'q': 
            quit = True 
            good_input = True
"""    
