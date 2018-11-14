from Pokemon import *
from NeuralNetwork import *
import random
import numpy as np
import matplotlib.pyplot as plt

# 2 Pokemon, randomly chosen, both controlled by the QNN, state is unchanging
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


def Q_network():
    Q = NeuralNetwork((state_length, 3 * state_length, action_length), sigmoid)  # will a single layer work? maybe???
    # epsilon: exploration v eploitation factor
    e = 5.0
    for epoch in range(1000):
        # Initialize battle
    #    print("New Battle")
        p = Pokemon(random.choice(types), "Q", Q)  #random.choice(types)
        q = Pokemon(random.choice(types), "Q", Q)
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

            # Mask illegal moves, state holds this too...
            faster_attack_mask = np.array([1.0 if a in faster.attacks else 0.0 for a in attacks.keys()])
            slower_attack_mask = np.array([1.0 if a in slower.attacks else 0.0 for a in attacks.keys()])

            # Get Q values with noise
            faster_attack_Q = Q.feed(faster_state)
            faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
            faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
    #        print(faster.type, np.multiply(faster_attack_Q, faster_attack_mask))
            slower_attack_Q = Q.feed(slower_state)
            slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
            slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
    #        print(slower.type, np.multiply(slower_attack_Q, slower_attack_mask))

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
            faster_reward = slower_HP_lost
            if slower.stats["HP"] > 0:
                faster_HP_lost = slower.attack(faster, slower_attack)
                faster_reward -= faster_HP_lost
                # Q(s,a) = r + γ(max(Q(s’,a’))
                slower_reward = (faster_HP_lost - slower_HP_lost)
                slower_attack_Q[slower_attack_index] = slower_reward
                Q.train_simple(state=slower_state, outcome=np.multiply(slower_attack_Q, slower_attack_mask))
            faster_attack_Q[faster_attack_index] = faster_reward
            Q.train_simple(state=faster_state, outcome=np.multiply(faster_attack_Q, faster_attack_mask))

        e *= 0.995 # too high and it won't explore with the even matchups
    for t1 in types:
        for t2 in types:
            print(t1, t2)
            print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))


def Q_Network_elu():
    # 2 Pokemon, randomly chosen, both controlled by the QNN, using elu activation (lr = 0.01)
    print("Q network replay")
    Q = NeuralNetwork((state_length, 5 * state_length, 2 * state_length, state_length, action_length), elu)  # will a single layer work? maybe???
    # epsilon: exploration v eploitation factor
    e = 5.0
    for epoch in range(1000):
        # Initialize battle
        p = Pokemon(random.choice(types), "Q", Q)
        q = Pokemon(random.choice(types), "Q", Q)
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

            # Get states
            faster_state = get_state(faster, slower)
            slower_state = get_state(slower, faster)

            # Mask illegal moves
            faster_attack_mask = faster_state[-4:]
            slower_attack_mask = slower_state[-4:]

            # Get Q values with noise
            faster_attack_Q = Q.feed(faster_state)
            faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
            faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
            faster_attack_Q_with_noise[faster_attack_Q_with_noise==0] = -1

            slower_attack_Q = Q.feed(slower_state)
            slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
            slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
            slower_attack_Q_with_noise[slower_attack_Q_with_noise==0] = -1

            # Get name of attack
            faster_attack_index = np.argmax(faster_attack_Q_with_noise)
            faster_attack = list(attacks.keys())[faster_attack_index]
            slower_attack_index = np.argmax(slower_attack_Q_with_noise)
            slower_attack = list(attacks.keys())[slower_attack_index]

            # Perform action, get reward, update
            slower_HP_lost = faster.attack(slower, faster_attack)
            # Q(s,a) = r + γ(max(Q(s’,a’))
            # γ(max(Q(s’,a’))
            faster_reward = slower_HP_lost
            if slower.stats["HP"] > 0:
                faster_HP_lost = slower.attack(faster, slower_attack)
                #faster_reward -= faster_HP_lost
                # Q(s,a) = r + γ(max(Q(s’,a’))
                slower_reward = faster_HP_lost #- slower_HP_lost
                slower_attack_Q[slower_attack_index] = slower_reward
                Q.train_simple(state=slower_state, outcome=slower_attack_Q)
            faster_attack_Q[faster_attack_index] = faster_reward
            Q.train_simple(state=faster_state, outcome=faster_attack_Q)

        e *= 0.995 # too high and it won't explore with the even matchups

    for t1 in types:
        for t2 in types:
            print(t1, t2)
            print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))


def Q_Network_elu_replay():
    # 2 Pokemon, randomly chosen, both controlled by the QNN, trained only by experience replay (lr = 0.00001)
    print("Q network replay")
    Q = NeuralNetwork((state_length, 2*state_length, state_length, action_length), elu)  # will a single layer work? maybe???
    # epsilon: exploration v eploitation factor
    e = 5.0
    # Memory: (state, action, reward)
    replay = []
    # How often to train
    every = 100
    for epoch in range(1500):
        # Initialize battle
        p = Pokemon(random.choice(types), "Q", Q)
        q = Pokemon(random.choice(types), "Q", Q)
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

            # Get states
            faster_state = get_state(faster, slower)
            slower_state = get_state(slower, faster)

            # Mask illegal moves
            faster_attack_mask = faster_state[-4:]
            slower_attack_mask = slower_state[-4:]

            # Get Q values with noise
            faster_attack_Q = Q.feed(faster_state)
            faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
            faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
            faster_attack_Q_with_noise[faster_attack_Q_with_noise==0] = -1

            slower_attack_Q = Q.feed(slower_state)
            slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
            slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
            slower_attack_Q_with_noise[slower_attack_Q_with_noise==0] = -1

            # Get name of attack
            faster_attack_index = np.argmax(faster_attack_Q_with_noise)
            faster_attack = list(attacks.keys())[faster_attack_index]
            slower_attack_index = np.argmax(slower_attack_Q_with_noise)
            slower_attack = list(attacks.keys())[slower_attack_index]

            # Perform action, get reward, update
            slower_HP_lost = faster.attack(slower, faster_attack)
            # Q(s,a) = r + γ(max(Q(s’,a’))
            # γ(max(Q(s’,a’))
            faster_reward = slower_HP_lost
            if slower.stats["HP"] > 0:
                faster_HP_lost = slower.attack(faster, slower_attack)
                #faster_reward -= faster_HP_lost
                # Q(s,a) = r + γ(max(Q(s’,a’))
                slower_reward = faster_HP_lost #- slower_HP_lost
                slower_attack_Q[slower_attack_index] = slower_reward
                replay += [(slower_state, slower_attack_index, slower_reward)]
            faster_attack_Q[faster_attack_index] = faster_reward
            replay += [(faster_state, faster_attack_index, faster_reward)]

        e *= 0.995 # too high and it won't explore with the even matchups
        if (epoch % every == (every - 1)):
            Q.train_replay_batch(replay, 100)

    for t1 in types:
        for t2 in types:
            print(t1, t2)
            print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))

def Q_Network_elu_during_and_replay():
    # 2 Pokemon, randomly chosen, both controlled by the QNN, trained by experience replay and during
    print("Q network replay")
    Q = NeuralNetwork((state_length, 2*state_length, state_length, action_length), elu)  # will a single layer work? maybe???
    # epsilon: exploration v eploitation factor
    e = 5.0
    # Memory: (state, action, reward)
    replay = []
    # Training param
    every = 100
    during_lr = 0.01
    batch_size = 100
    replay_lr = during_lr / batch_size
    for epoch in range(500):
        # Initialize battle
        p = Pokemon(random.choice(types), "Q", Q)
        q = Pokemon(random.choice(types), "Q", Q)
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

            # Get states
            faster_state = get_state(faster, slower)
            slower_state = get_state(slower, faster)

            # Mask illegal moves
            faster_attack_mask = faster_state[-4:]
            slower_attack_mask = slower_state[-4:]

            # Get Q values with noise
            faster_attack_Q = Q.feed(faster_state)
            faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
            faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
            faster_attack_Q_with_noise[faster_attack_Q_with_noise==0] = -1

            slower_attack_Q = Q.feed(slower_state)
            slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
            slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
            slower_attack_Q_with_noise[slower_attack_Q_with_noise==0] = -1

            # Get name of attack
            faster_attack_index = np.argmax(faster_attack_Q_with_noise)
            faster_attack = list(attacks.keys())[faster_attack_index]
            slower_attack_index = np.argmax(slower_attack_Q_with_noise)
            slower_attack = list(attacks.keys())[slower_attack_index]

            # Perform action, get reward, update
            slower_HP_lost = faster.attack(slower, faster_attack)
            # Q(s,a) = r + γ(max(Q(s’,a’))
            # γ(max(Q(s’,a’))
            faster_reward = slower_HP_lost
            if slower.stats["HP"] > 0:
                faster_HP_lost = slower.attack(faster, slower_attack)
                #faster_reward -= faster_HP_lost
                # Q(s,a) = r + γ(max(Q(s’,a’))
                slower_reward = faster_HP_lost #- slower_HP_lost
                slower_attack_Q[slower_attack_index] = slower_reward
                slower_attack_Q = np.multiply(slower_attack_mask, slower_attack_Q)
                slower_attack_Q[slower_attack_Q==0] = -1
                replay += [(slower_state, slower_attack_index, slower_reward)]
                Q.train_simple(state=slower_state, outcome=slower_attack_Q, lr=during_lr)
            faster_attack_Q[faster_attack_index] = faster_reward
            faster_attack_Q = np.multiply(faster_attack_mask, faster_attack_Q)
            faster_attack_Q[faster_attack_Q==0] = -1
            replay += [(faster_state, faster_attack_index, faster_reward)]
            Q.train_simple(state=faster_state, outcome=faster_attack_Q, lr=during_lr)


        e *= 0.995 # too high and it won't explore with the even matchups
        if (epoch % every == (every - 1)):
            Q.train_replay_batch(replay, batch_size, lr=replay_lr) # epoch = 100

    for t1 in types:
        for t2 in types:
            print(t1, t2)
            print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))

# Separate target network:Q-Target = r + γQ(s’,argmax_a(Q(s’,a,ϴ));ϴ-), where ϴ- is a previous ϴ. occassionally/slowly, ϴ- = ϴ. Tau annealing parameter.
# Double DQN: Q-Target = r + γQ(s’,argmax_a(Q(s’,a,ϴ));ϴ’), switch roles of ϴ and ϴ' everytime
# Dueling DQN: Q(s, a) = A(s, a) + V(s), helps with the problem of no actions preventing death. Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a')) (therefore, mean(Q(s,a)) = V(s))
# Prioritized Experience Replay: P(i) = p(i)**α / sum_n(p(n)**α); p(i) = |error| + ε or p(i) = 1/rank(i), rank(i) = index(sort(i, key=|error|)) AND Δw = Δw/(N*P(i))**b, b=0->1
# Curiosity: min_(ϴP,ϴI,ϴF) [-λπ(st;θP) + (1 − β)LI + βLF], where the three elements are the policy (maximize total reward), the inverse model (minimize error predicting which action will cause a certain state change) and the forward model (minimize error predicting the next state given the current state and taken action.

def Q_Network_discounted_future():
    # 2 Pokemon, randomly chosen, both controlled by the QNN, trained by experience replay and during, using discounted future rewards.
    print("Q network discount")
    Q = NeuralNetwork((state_length, state_length, action_length), elu)
    # Epsilon: exploration v eploitation factor
    e = 5.0
    # Future discount
    y = 0.9 # higher value, harder to train (bias due to maximum?) with other params, 0.9 is max. longer train time if higher
    # Memory: (state, action, reward)
    replay = []
    # Training param
    every = 100
    during_lr = 0.01
    batch_size = 100
    replay_lr = during_lr / batch_size
    for epoch in range(2000):
        # Initialize battle
        p = Pokemon(random.choice(types), "Q", Q)
        q = Pokemon(random.choice(types), "Q", Q)
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

            # Get states
            faster_state = get_state(faster, slower)
            slower_state = get_state(slower, faster)

            # Mask illegal moves
            faster_attack_mask = faster_state[-4:]
            slower_attack_mask = slower_state[-4:]

            # Get Q values with noise
            faster_attack_Q = Q.feed(faster_state)
            faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
            faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
            faster_attack_Q_with_noise[faster_attack_Q_with_noise==0] = -1

            slower_attack_Q = Q.feed(slower_state)
            slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
            slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
            slower_attack_Q_with_noise[slower_attack_Q_with_noise==0] = -1

            # Get name of attack
            faster_attack_index = np.argmax(faster_attack_Q_with_noise)
            faster_attack = list(attacks.keys())[faster_attack_index]
            slower_attack_index = np.argmax(slower_attack_Q_with_noise)
            slower_attack = list(attacks.keys())[slower_attack_index]

            # Perform action, get reward, update
            slower_HP_lost = faster.attack(slower, faster_attack)
            # Q(s,a) = r + γ(max(Q(s’,a’))
            # γ(max(Q(s’,a’))
            faster_reward = slower_HP_lost
            if slower.stats["HP"] > 0:
                faster_HP_lost = slower.attack(faster, slower_attack)
                #faster_reward -= faster_HP_lost
                # Q(s,a) = r + γ(max(Q(s’,a’))
                slower_reward = faster_HP_lost #- slower_HP_lost
                new_slow_Q = Q.feed(slower_state)
                future_slow_reward = np.multiply(new_slow_Q, slower_attack_mask)
                future_slow_reward[future_slow_reward==0] = -1
                future_slow_reward_index = np.argmax(future_slow_reward)
                slower_attack_Q[slower_attack_index] = slower_reward + y * new_slow_Q[future_slow_reward_index]
                slower_attack_Q = np.multiply(slower_attack_mask, slower_attack_Q)
                slower_attack_Q[slower_attack_Q==0] = -1
                replay += [(slower_state, slower_attack_index, slower_attack_Q[slower_attack_index])]
                Q.train_simple(state=slower_state, outcome=slower_attack_Q, lr=during_lr)
            new_fast_Q = Q.feed(faster_state)
            future_fast_reward = np.multiply(new_fast_Q, faster_attack_mask)
            future_fast_reward[future_fast_reward==0] = -1
            future_fast_reward_index = np.argmax(future_fast_reward)
            faster_attack_Q[faster_attack_index] = faster_reward + y * new_fast_Q[future_fast_reward_index]
            print("reward", faster_attack_Q[faster_attack_index])
            faster_attack_Q = np.multiply(faster_attack_mask, faster_attack_Q)
            faster_attack_Q[faster_attack_Q==0] = -1
            replay += [(faster_state, faster_attack_index, faster_attack_Q[faster_attack_index])]
            Q.train_simple(state=faster_state, outcome=faster_attack_Q, lr=during_lr)

        e *= 0.995 # too high and it won't explore with the even matchups
        if (epoch % every == (every - 1)):
            Q.train_replay_batch(replay, batch_size, epoch=10, lr=replay_lr)
            replay = []

    for t1 in types:
        for t2 in types:
            print(t1, t2)
            print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))

def Q_Network_swords_dance():
    # 2 Pokemon, random, QNN controlled, trained by both, DFR and given swords dance. Not tested.
    print("Q network swords dance")
    Q = NeuralNetwork((state_length, 2 * state_length, state_length, action_length), elu)
    # Epsilon: exploration v eploitation factor
    e = 5.0
    # Future discount
    y = 0.9 # higher value, harder to train (bias due to maximum?) with other params, 0.9 is max. longer train time if higher
    # Memory: (state, action, reward)
    replay = []
    # Training param
    every = 100
    during_lr = 0.01
    batch_size = 100
    replay_lr = during_lr / batch_size
    for epoch in range(5000):
        # Initialize battle
        p = Pokemon(random.choice(types), "Q", Q)
        q = Pokemon(random.choice(types), "Q", Q)
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

            # Get states
            faster_state = get_state(faster, slower)
            slower_state = get_state(slower, faster)

            # Mask illegal moves
            faster_attack_mask = faster_state[-4:]
            slower_attack_mask = slower_state[-4:]

            # Get Q values with noise
            faster_attack_Q = Q.feed(faster_state)
            faster_attack_Q_with_noise = faster_attack_Q if e < random.random() else np.random.random(action_length)
            faster_attack_Q_with_noise = np.multiply(faster_attack_Q_with_noise, faster_attack_mask)
            faster_attack_Q_with_noise[faster_attack_Q_with_noise==0] = -1

            slower_attack_Q = Q.feed(slower_state)
            slower_attack_Q_with_noise = slower_attack_Q if e < random.random() else np.random.random(action_length)
            slower_attack_Q_with_noise = np.multiply(slower_attack_Q_with_noise, slower_attack_mask)
            slower_attack_Q_with_noise[slower_attack_Q_with_noise==0] = -1

            # Get name of attack
            faster_attack_index = np.argmax(faster_attack_Q_with_noise)
            faster_attack = list(attacks.keys())[faster_attack_index]
            slower_attack_index = np.argmax(slower_attack_Q_with_noise)
            slower_attack = list(attacks.keys())[slower_attack_index]

            # Perform action, get reward, update
            slower_HP_lost = faster.attack(slower, faster_attack)
            # Q(s,a) = r + γ(max(Q(s’,a’))
            # γ(max(Q(s’,a’))
            faster_reward = slower_HP_lost
            if slower.stats["HP"] > 0:
                faster_HP_lost = slower.attack(faster, slower_attack)
                #faster_reward -= faster_HP_lost
                # Q(s,a) = r + γ(max(Q(s’,a’))
                slower_reward = faster_HP_lost #- slower_HP_lost
                new_slow_Q = Q.feed(slower_state)
                future_slow_reward = np.multiply(new_slow_Q, slower_attack_mask)
                future_slow_reward[future_slow_reward==0] = -1
                future_slow_reward_index = np.argmax(future_slow_reward)
                slower_attack_Q[slower_attack_index] = slower_reward + y * new_slow_Q[future_slow_reward_index]
                slower_attack_Q = np.multiply(slower_attack_mask, slower_attack_Q)
                slower_attack_Q[slower_attack_Q==0] = -1
                replay += [(slower_state, slower_attack_index, slower_attack_Q[slower_attack_index])]
                Q.train_simple(state=slower_state, outcome=slower_attack_Q, lr=during_lr)
            new_fast_Q = Q.feed(faster_state)
            future_fast_reward = np.multiply(new_fast_Q, faster_attack_mask)
            future_fast_reward[future_fast_reward==0] = -1
            future_fast_reward_index = np.argmax(future_fast_reward)
            faster_attack_Q[faster_attack_index] = faster_reward + y * new_fast_Q[future_fast_reward_index]
            print("reward", faster_attack_Q[faster_attack_index])
            faster_attack_Q = np.multiply(faster_attack_mask, faster_attack_Q)
            faster_attack_Q[faster_attack_Q==0] = -1
            replay += [(faster_state, faster_attack_index, faster_attack_Q[faster_attack_index])]
            Q.train_simple(state=faster_state, outcome=faster_attack_Q, lr=during_lr)

        e *= 0.995 # too high and it won't explore with the even matchups
        if (epoch % every == (every - 1)):
            Q.train_replay_batch(replay, batch_size, epoch=10, lr=replay_lr)
            replay = []

    for t1 in types:
        for t2 in types:
            print(t1, t2)
            print(Q.feed(get_state(Pokemon(t1), Pokemon(t2))))


# Separate target network:Q-Target = r + γQ(s’,argmax_a(Q(s’,a,ϴ));ϴ-), where ϴ- is a previous ϴ. occassionally/slowly, ϴ- = ϴ. Tau annealing parameter.
# Double DQN: Q-Target = r + γQ(s’,argmax_a(Q(s’,a,ϴ));ϴ’), switch roles of ϴ and ϴ' everytime
# Dueling DQN: Q(s, a) = A(s, a) + V(s), helps with the problem of no actions preventing death. Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a')) (therefore, mean(Q(s,a)) = V(s))
# Prioritized Experience Replay: P(i) = p(i)**α / sum_n(p(n)**α); p(i) = |error| + ε or p(i) = 1/rank(i), rank(i) = index(sort(i, key=|error|)) AND Δw = Δw/(N*P(i))**b, b=0->1
# Curiosity: min_(ϴP,ϴI,ϴF) [-λπ(st;θP) + (1 − β)LI + βLF], where the three elements are the policy (maximize total reward), the inverse model (minimize error predicting which action will cause a certain state change) and the forward model (minimize error predicting the next state given the current state and taken action.


if __name__ == "__main__":
    #Q_network()
    #Q_Network_elu()
    #Q_Network_elu_replay()
    #Q_Network_elu_during_and_replay()
    #Q_Network_discounted_future()
    Q_Network_swords_dance()