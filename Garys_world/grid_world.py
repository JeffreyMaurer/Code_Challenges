import numpy as np
import random as r
import pygame as pg
import sys
from NeuralNetwork import *

np.random.seed(20)
r.seed(1)

values = np.zeros((5,5),int)
values[values == 0] = -1

x = np.random.choice(np.arange(5),2)
y = np.random.choice(np.arange(5),2)

values[x,y] = 10
values[1,0] = 10

x = r.randrange(0,5)
y = r.randrange(0,5)

values[x,y] = 1000

x = np.random.choice(np.arange(5),3)
y = np.random.choice(np.arange(5),3)

values[x,y] = -1000000

values[0,0] = 0

pg.init()

size = width, height = 600, 500
black = (0,0,0)
white = (255,255,255)
light_red = (255,200,200)
red = (255,0,0)
green = (0,255,0)
light_green = (200,255,200)

screen = pg.display.set_mode(size)
pg.display.set_caption("Grid World")
font = pg.font.SysFont(None, 100)
score = 0
grid = {}
pos = {}
for i in range(5):
    for j in range(5):
        grid[(i,j)] = white
        pos[(i,j)] = [10 + 600*i/5, 100 + 400 * j/5, 600 / 6, 400 / 6]

agent = pg.image.load("agent.png")
agent_rect = agent.get_rect()
agent_pos = (0,0)
NN = True
color = True
table = False
state_length = 25
action_length = 5
Q = NeuralNetwork((state_length, state_length, action_length), elu)
Q_table = {}
for i in range(5):
    for j in range(5):
        Q_table[(i,j)] = []
        for a in range(action_length):
            Q_table[(i,j)] += [0]
e = 1
lr = 0.0001
reward = {-1000000:0,-1:0,10:0.001,1000:100,0:0}
replay = []
epoch = 0
if table == False:
    while 1:
        epoch += 1
        clock = pg.time.Clock()
        #clock.tick(60)
        state = np.zeros(25)
        state[agent_pos[0]*5+agent_pos[1]] = 1
        choice = Q.feed(state)
        a = 0
        if NN == False:
            for event in pg.event.get():
                if event.type == pg.QUIT: sys.exit()
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT and agent_pos[0] > 0:
                        agent_pos = (agent_pos[0] - 1, agent_pos[1])
                    if event.key == pg.K_RIGHT and agent_pos[0] < 4:
                        agent_pos = (agent_pos[0] + 1, agent_pos[1])
                    if event.key == pg.K_UP and agent_pos[1] > 0:
                        agent_pos = (agent_pos[0], agent_pos[1] - 1)
                    if event.key == pg.K_DOWN and agent_pos[1] < 4:
                        agent_pos = (agent_pos[0], agent_pos[1] + 1)
        else:
            if r.random() > e:
                a = np.argmax(choice)
            else:
                a = r.choice([0,1,2,3])
            if a == 0 and agent_pos[0] > 0:
                agent_pos = (agent_pos[0] - 1, agent_pos[1])
            if a == 1 and agent_pos[0] < 4:
                agent_pos = (agent_pos[0] + 1, agent_pos[1])
            if a == 2 and agent_pos[1] > 0:
                agent_pos = (agent_pos[0], agent_pos[1] - 1)
            if a == 3 and agent_pos[1] < 4:
                agent_pos = (agent_pos[0], agent_pos[1] + 1)
        new_state = np.zeros(25)
        new_state[agent_pos[0]*5+agent_pos[1]] = 1

        agent_rect.x, agent_rect.y = pos[agent_pos][0:2]

        if color == True:
            if values[agent_pos[0],agent_pos[1]] == -1:
                grid[agent_pos] = light_red
            if values[agent_pos[0],agent_pos[1]] == -1000000:
                grid[agent_pos] = red
            if values[agent_pos[0],agent_pos[1]] == 10:
                grid[agent_pos] = light_green
            if values[agent_pos[0],agent_pos[1]] == 1000:
                grid[agent_pos] = green

        score += values[agent_pos[0],agent_pos[1]]
        end = False
        if values[agent_pos[0],agent_pos[1]] == -1000000:
            agent_pos = (0,0)
            agent_rect.x, agent_rect.y = pos[agent_pos][0:2]

        if NN == True:
            choice[a] = reward[values[agent_pos[0],agent_pos[1]]] + 0.9*np.max(Q.feed(new_state))
            replay += [(state, a, reward[values[agent_pos[0],agent_pos[1]]], new_state)]
            Q.train_simple(state,choice,lr=0.01)

            if len(replay) > 1000:
                Q.train_replay_batch(replay,100,epoch=10,lr=lr/100)
            if (epoch % 100 == 0):
                e *= 0.98
                print(e)

        screen.fill(black)
        for i in range(5):
            for j in range(5):
                pg.draw.rect(screen, grid[(i,j)], pos[(i,j)])
        screen.blit(font.render(str(score), True, pg.Color(255,255,255)), [0, 0])
        screen.blit(agent, agent_rect)
        pg.display.flip()
else:
    while 1:
        epoch += 1
        clock = pg.time.Clock()
        #clock.tick(60)
        state = (agent_pos[0], agent_pos[1])
        a = 0
        if r.random() > e:
            a = Q_table[state].index(max(Q_table[state]))
        else:
            a = r.choice([0,1,2,3])
        if a == 0 and agent_pos[0] > 0:
            agent_pos = (agent_pos[0] - 1, agent_pos[1])
        if a == 1 and agent_pos[0] < 4:
            agent_pos = (agent_pos[0] + 1, agent_pos[1])
        if a == 2 and agent_pos[1] > 0:
            agent_pos = (agent_pos[0], agent_pos[1] - 1)
        if a == 3 and agent_pos[1] < 4:
            agent_pos = (agent_pos[0], agent_pos[1] + 1)
        new_state = agent_pos

        agent_rect.x, agent_rect.y = pos[agent_pos][0:2]

        if color == True:
            if values[agent_pos[0],agent_pos[1]] == -1:
                grid[agent_pos] = light_red
            if values[agent_pos[0],agent_pos[1]] == -1000000:
                grid[agent_pos] = red
            if values[agent_pos[0],agent_pos[1]] == 10:
                grid[agent_pos] = light_green
            if values[agent_pos[0],agent_pos[1]] == 1000:
                grid[agent_pos] = green

        score += values[agent_pos[0],agent_pos[1]]
        if values[agent_pos[0],agent_pos[1]] == -1000000:
            agent_pos = (0,0)
            agent_rect.x, agent_rect.y = pos[agent_pos][0:2]
        print(state)
        print(new_state)
        Q_table[state][a] = Q_table[state][a] + 0.001*(reward[values[agent_pos[0],agent_pos[1]]] + 0.9*max(Q_table[new_state]) - Q_table[state][a])

        if (epoch % 100 == 0):
            e *= 0.99
            print(e)

        screen.fill(black)
        for i in range(5):
            for j in range(5):
                pg.draw.rect(screen, grid[(i,j)], pos[(i,j)])
        screen.blit(font.render(str(score), True, pg.Color(255,255,255)), [0, 0])
        screen.blit(agent, agent_rect)
        pg.display.flip()