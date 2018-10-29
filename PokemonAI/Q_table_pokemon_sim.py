from Pokemon import *
import random

# 2 Pokemon, prechosen, one choose moves by random, one by Q matrix
print("Q")
# Q matrix, actions by Estates, by Pstates
# Estates: enemy green, enemy red, enemy blue
# Pstates: player green, player red, player blue
# Actions: tackle or stab
player_type = ["RED", "GREEN", "BLUE"]
enemy_type = ["RED", "GREEN", "BLUE"]
actions = ["TACKLE", "SANGUINE"]
Q = {}
state_values = [(e, p) for e in enemy_type for p in player_type]
for state in state_values:
    Q[state] = {}
    for a in actions:
        Q[state][a] = 0
# Train Q
record = []
epsilon = 1.0
print(Q[state])
for epoch in range(0):
    # Initialize battle
    p = Pokemon("RED", "Q", Q)
    q = Pokemon("RED", "TACKLE")
    faster, slower = (p, q) if (p.stats["SPD"] > q.stats["SPD"]) else (q, p)
    state = (q.type, p.type)
    # Play
    while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
        print(Q[state])
        # Make a choice of action
        fa, sa = (faster.choose_attack(state=state, epsilon=epsilon),
                    slower.choose_attack(state=state, epsilon=epsilon))
        reward = faster.attack(slower, fa)
        print(fa, sa)
        # Get reward
        if faster is q:
            reward = -reward
        print("damage received", reward)
        Q[state][sa] += reward  # state hasn't changed...
        if slower.stats["HP"] == 0:
            break
        reward = slower.attack(faster, sa)
        # Get reward
        if slower is q:
            reward = -reward
        print("damage dealt", reward)
        Q[state][sa] += reward  # state hasn't changed...
    # Keep track of wins and losses
    if (q.stats["HP"] == 0):
        record.append("WIN")
    else:
        record.append("LOS")
    epsilon -= 0.1
    print(Q[state])
print(record)
# Does not converge if not given enough time to win a game...


# 2 Pokemon, prechosen, both controlled by the NN
print("2Q")
player_type = ["RED", "GREEN", "BLUE"]
enemy_type = ["RED", "GREEN", "BLUE"]
# actions will be a list of moves, but each poke will only be able to choose
# one of their own!!
actions = list(attacks.keys())
Q = {}
state_values = [(e, p) for e in enemy_type for p in player_type]
for state in state_values:
    Q[state] = {}
    for a in actions:
        Q[state][a] = 0
# Train Q
record = []
epsilon = 1.0
print(Q)
for epoch in range(0):
    # Initialize battle
    p = Pokemon("BLUE", "Q", Q)
    q = Pokemon("BLUE", "Q", Q)
    faster, slower = (p, q) if (p.stats["SPD"] > q.stats["SPD"]) else (q, p)
    state = (q.type, p.type)
    # Play
    while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
        print(Q[state])
        # Make a choice of action
        fa, sa = (faster.choose_attack(state=state, epsilon=epsilon),
                    slower.choose_attack(state=state, epsilon=epsilon))
        freward = faster.attack(slower, fa)
        print(fa, sa)
        # Get reward
        Q[(slower.type, faster.type)][fa] += freward  # state hasn't changed...
        Q[(faster.type, slower.type)][sa] -= freward  # state hasn't changed...
        if slower.stats["HP"] == 0:
            break
        sreward = slower.attack(faster, sa)
        # Get reward
        Q[(faster.type, slower.type)][sa] += sreward  # state hasn't changed...
        Q[(slower.type, faster.type)][fa] -= sreward  # state hasn't changed...
    epsilon -= 0.1
    print(Q[state])

# 2 Pokemon, randomly chosen, both controlled by the Q, STAB and effectiveness implemented
print("3Q")
player_type = ["RED", "GREEN", "BLUE"]
enemy_type = ["RED", "GREEN", "BLUE"]
actions = list(attacks.keys())
Q = {}
state_values = [(e, p) for e in enemy_type for p in player_type]
for state in state_values:
    Q[state] = {}
    for a in actions:
        Q[state][a] = 0
# Train Q
record = []
e = 1.0
print(Q)
for epoch in range(0):
    # Initialize battle
    p = Pokemon(random.choice(types), "Q", Q)
    q = Pokemon(random.choice(types), "Q", Q)
    print("Current actors", q.type, p.type)
    print("Q of", p.type, Q[(q.type, p.type)])
    print("Q of", q.type, Q[(p.type, q.type)])
    # Play
    while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
        # Turn order
        faster, slower = None, None
        if (p.stats["SPD"] > q.stats["SPD"]):
            faster, slower = p, q
        elif (p.stats["SPD"] < q.stats["SPD"]):
            faster, slower = q, p
        else:
            faster, slower = (q, p) if random.random() > 0.5 else (p, q)
        # Make a choice of action
        fa, sa = (faster.choose_attack(state=(slower.type, faster.type), epsilon=e),
                  slower.choose_attack(state=(faster.type, slower.type), epsilon=e))
        print(faster.type, slower.type)
        print(fa, sa)
        # Perform action
        freward = faster.attack(slower, fa)
        # Get reward
        Q[(slower.type, faster.type)][fa] += freward  # state hasn't changed...
        # Q[(faster.type, slower.type)][sa] -= freward  # state hasn't changed... # No penalty for a move that wasn't able to be made
        if slower.stats["HP"] == 0:
            break
        Q[(faster.type, slower.type)][sa] -= freward
        # Perform action
        sreward = slower.attack(faster, sa)
        # Get reward
        Q[(faster.type, slower.type)][sa] += sreward  # state hasn't changed...
        Q[(slower.type, faster.type)][fa] -= sreward  # state hasn't changed...
    e -= 0.01 # too high and it won't explore with the even matchups
    if p.stats["HP"] == 0:
        print(q.type, "wins")
    else:
        print(p.type, "wins")
    print("Q of ", p.type, Q[(q.type, p.type)])
    print("Q of ", q.type, Q[(p.type, q.type)])

for matchup in Q:
    print(matchup)
    print(Q[matchup])
# The model can't learn if it can't win... no matter what (Orbitz!)

# 2 Pokemon, randomly chosen, both controlled by the Q, STAB and effectiveness implemented
# Some other changes... did not check if it works...
print("4Q")
player_type = ["RED", "GREEN", "BLUE"]
enemy_type = ["RED", "GREEN", "BLUE"]
Q = {}
state_values = [(e, p) for e in enemy_type for p in player_type]
for state in state_values:
    Q[state] = {}
    for a in ["TACKLE", base_attacks[state[1]]]:
        Q[state][a] = 0
# Train Q
record = []
e = 1.0
y = 0.9
print(Q)
for epoch in range(1000):
    # Initialize battle
    p = Pokemon(random.choice(types), "Q", Q)
    q = Pokemon(random.choice(types), "Q", Q)
    print("Current actors", q.type, p.type)
    print("Q of", p.type, Q[(q.type, p.type)])
    print("Q of", q.type, Q[(p.type, q.type)])
    # Play
    while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
        # Turn order
        faster, slower = None, None
        if (p.stats["SPD"] > q.stats["SPD"]):
            faster, slower = p, q
        elif (p.stats["SPD"] < q.stats["SPD"]):
            faster, slower = q, p
        else:
            faster, slower = (q, p) if random.random() > 0.5 else (p, q)
        # Make a choice of action
        fa, sa = (faster.choose_attack(state=(slower.type, faster.type), epsilon=e),
                  slower.choose_attack(state=(faster.type, slower.type), epsilon=e))
        print(faster.type, slower.type)
        print(fa, sa)
        # Perform action
        freward = faster.attack(slower, fa)
        # Get reward
        Q[(slower.type, faster.type)][fa] += freward #+ y * max(Q[(slower.type, faster.type)], key=Q[(slower.type, faster.type)].get)# state hasn't changed...
        # Q[(faster.type, slower.type)][sa] -= freward  # state hasn't changed... # No penalty for a move that wasn't able to be made
        if slower.stats["HP"] == 0:
            break
        # Perform action
        sreward = slower.attack(faster, sa)
        # Get reward
        Q[(faster.type, slower.type)][sa] += sreward - freward # state hasn't changed...
        Q[(slower.type, faster.type)][fa] -= sreward  # state hasn't changed...
    e *= 0.99 # too high and it won't explore with the even matchups
    if p.stats["HP"] == 0:
        print(q.type, "wins")
    else:
        print(p.type, "wins")
    print("Q of ", p.type, Q[(q.type, p.type)])
    print("Q of ", q.type, Q[(p.type, q.type)])

for matchup in Q:
    print(matchup)
    print(Q[matchup])

