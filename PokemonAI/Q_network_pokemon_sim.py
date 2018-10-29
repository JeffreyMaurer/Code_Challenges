from Pokemon import *
from NeuralNetwork import *
import random
import numpy as np

# 2 Pokemon, randomly chosen, both controlled by the QNN
print("Q network")
player_type = ["RED", "GREEN", "BLUE"]
enemy_type = ["RED", "GREEN", "BLUE"]
state_length = len(player_type) + len(enemy_type) + len(attacks)  # 10
action_length = len(attacks)  # 4, some are illegal though, need to train that out of the machine??? not necessarily. dont.

# Easy function to get the state into a workable form for each player
def get_state(player, enemy):
    player_state = [(1 if player.type == t else 0) for t in types]
    player_state += [(1 if enemy.type == t else 0) for t in types]
    player_state += [(1 if move in player.attacks else 0) for move in attacks]
    return(np.array(player_state))

Q = NeuralNetwork((state_length, 3 * state_length, action_length))  # will a single layer work? maybe???
# epsilon: exploration v eploitation factor
e = 2.0
# gamma: future reward discount factor
y = 0.9
# memories to replay
replay = []
print("BLUE", Q.feed(get_state(Pokemon("BLUE"), Pokemon("RED"))))
print("RED", Q.feed(get_state(Pokemon("RED"), Pokemon("BLUE"))))
for epoch in range(1):
    # Initialize battle
#    print("New Battle")
    p = Pokemon("BLUE", "Q", Q)  #random.choice(types)
    q = Pokemon("RED", "Q", Q)
#    print("Current actors", q.type, p.type)
    # Play
    while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
#        print("New Round")
        # Turn order
        faster, slower = None, None
        if (p.stats["SPD"] > q.stats["SPD"]):
            faster, slower = p, q
        elif (p.stats["SPD"] < q.stats["SPD"]):
            faster, slower = q, p
        else:
            faster, slower = (q, p) if random.random() > 0.5 else (p, q)

        # Get states
        faster_state = get_state(faster, slower)
        slower_state = get_state(slower, faster)

        # Mask illegal moves
        faster_attack_mask = np.array([1.0 if a in faster.attacks else 0.0 for a in attacks.keys()])
        slower_attack_mask = np.array([1.0 if a in slower.attacks else 0.0 for a in attacks.keys()])

        # Get Q values with noise
        faster_attack_Q = Q.feed(faster_state)
        faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
        faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
#        print(np.multiply(faster_attack_Q, faster_attack_mask))
        slower_attack_Q = Q.feed(slower_state)
        slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
        slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
#        print(np.multiply(slower_attack_Q, slower_attack_mask))

        # Get name of attack
        faster_attack_index = np.argmax(faster_attack_Q_with_noise)
        faster_attack = list(attacks.keys())[faster_attack_index]
        slower_attack_index = np.argmax(slower_attack_Q_with_noise)
        slower_attack = list(attacks.keys())[slower_attack_index]

#        print(faster.type, slower.type)
#        print(faster_attack, slower_attack)

        # Perform action, get reward, update
        slower_HP_lost = faster.attack(slower, faster_attack)
        # Q(s,a) = r + γ(max(Q(s’,a’))
        # γ(max(Q(s’,a’))
        Q_prime_fast = np.multiply(Q.feed(faster_state), faster_attack_mask)
        faster_delta_Q = slower_HP_lost # + y * max(Q_prime_fast)
        if slower.stats["HP"] > 0:
            faster_HP_lost = slower.attack(faster, slower_attack)
            faster_delta_Q -= faster_HP_lost
            # Q(s,a) = r + γ(max(Q(s’,a’))
            Q_prime_slow = np.multiply(Q.feed(slower_state), slower_attack_mask)
            slower_delta_Q = (faster_HP_lost - slower_HP_lost) # + y * max(Q_prime_slow)
            slower_attack_Q[slower_attack_index] += slower_delta_Q
            Q.train_simple(state=slower_state, outcome=slower_attack_Q)
#            print(slower_attack, slower_delta_Q)
#        print(faster_attack, faster_delta_Q)
        faster_attack_Q[faster_attack_index] += faster_delta_Q
        Q.train_simple(state=faster_state, outcome=faster_attack_Q)

    e *= 0.995 # too high and it won't explore with the even matchups
"""
    if p.stats["HP"] == 0:
        print(q.type, "wins")
    else:
        print(p.type, "wins")
"""
print("action")
"""
for t1 in types:
    for t2 in types:
        print(t1, t2)
        print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))
"""
print("BLUE", Q.feed(get_state(Pokemon("BLUE"), Pokemon("RED"))))
print("RED", Q.feed(get_state(Pokemon("RED"), Pokemon("BLUE"))))

"""
# 2 Pokemon, randomly chosen, both controlled by the QNN, trained only by random replay
print("Q network replay")
player_type = ["RED", "GREEN", "BLUE"]
enemy_type = ["RED", "GREEN", "BLUE"]
state_length = len(player_type) + len(enemy_type) + len(attacks)  # 10
action_length = len(attacks)  # 4, some are illegal though, need to train that out of the machine??? not necessarily. dont.
Q = NeuralNetwork((state_length, state_length, action_length), leaky_relu)  # will a single layer work? maybe???
for t1 in types:
    for t2 in types:
        print(t1, t2)
        print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))
# epsilon: exploration v eploitation factor
e = 2.0
# gamma: future reward discount factor
y = 0.9
# memories to replay
replay = []
attack_freq = np.array([0 for _ in attacks])
for epoch in range(0):
    # Initialize battle
    print("New Battle")
    p = Pokemon(random.choice(types), "Q", Q)  #random.choice(types)
    q = Pokemon(random.choice(types), "Q", Q)
    print("Current actors", q.type, p.type)
    # Play
    while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
        print("New Round")
        # Turn order
        faster, slower = None, None
        if (p.stats["SPD"] > q.stats["SPD"]):
            faster, slower = p, q
        elif (p.stats["SPD"] < q.stats["SPD"]):
            faster, slower = q, p
        else:
            faster, slower = (q, p) if random.random() > 0.5 else (p, q)

        # Get states
        faster_state = get_state(faster, slower)
        slower_state = get_state(slower, faster)

        # Mask illegal moves
        faster_attack_mask = np.array([1.0 if a in faster.attacks else 0.0 for a in attacks.keys()])
        slower_attack_mask = np.array([1.0 if a in slower.attacks else 0.0 for a in attacks.keys()])

        # Get Q values with noise
        faster_attack_Q = Q.feed(faster_state)
        faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
        faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
        print(np.multiply(faster_attack_Q, faster_attack_mask))
        slower_attack_Q = Q.feed(slower_state)
        slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
        slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
        print(np.multiply(slower_attack_Q, slower_attack_mask))

        # Get name of attack
        faster_attack_index = np.argmax(faster_attack_Q_with_noise)
        faster_attack = list(attacks.keys())[faster_attack_index]
        slower_attack_index = np.argmax(slower_attack_Q_with_noise)
        slower_attack = list(attacks.keys())[slower_attack_index]

        print(faster.type, slower.type)
        print(faster_attack, slower_attack)

        # Update attack_freq
        attack_freq[faster_attack_index] += 1
        attack_freq[slower_attack_index] += 1

        # Perform action, get reward, store
        slower_HP_lost = faster.attack(slower, faster_attack)
        if slower.stats["HP"] > 0:
            faster_HP_lost = slower.attack(faster, slower_attack)
        # s, a, r, s'
        replay.append((slower_state, slower_attack_index, faster_HP_lost - slower_HP_lost, slower_state))
        replay.append((faster_state, faster_attack_index, slower_HP_lost - faster_HP_lost, faster_state))
    if  ((epoch % 50) == 50):
        reps = random.sample(replay, 50)
        print(reps)
        Q.train_replay_batch(reps, attack_freq)

    e *= 0.995 # too high and it won't explore with the even matchups
    if p.stats["HP"] == 0:
        print(q.type, "wins")
    else:
        print(p.type, "wins")
for t1 in types:
    for t2 in types:
        print(t1, t2)
        print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))

# Two Pokemon Teams
p = [Pokemon("GREEN"), Pokemon("BLUE")]
q = [Pokemon("RED"), Pokemon("GREEN")]


def logic(team1, team2):
    pass


def view():
    pass


done = False
while (done):
    logic(p, q)
    view()

"""

