import random
import copy

types = ["RED", "BLUE", "GREEN"]

base_stats = {
"RED":
    {
    "HP": 100,
    "ATT": 100,
    "DEF": 100,
    "SPD": 100
    },
"BLUE":
    {
    "HP": 100,
    "ATT": 100,
    "DEF": 100,
    "SPD": 100
    },
"GREEN":
    {
    "HP": 100,
    "ATT": 100,
    "DEF": 100,
    "SPD": 100
    }
}

attacks = {
"TACKLE": {
    "POWER": 5,
    "TYPE": "NORMAL"
    },
"SANGUINE": {
    "POWER": 5,
    "TYPE": "RED"
    },
"NAVY": {
    "POWER": 5,
    "TYPE": "BLUE"
    },
"OLIVE": {
    "POWER": 5,
    "TYPE": "BLUE"
    }
}

base_attacks = {"RED": "SANGUINE",
                "BLUE": "NAVY",
                "GREEN": "OLIVE"}

class Pokemon:

    def __init__(self, type, brain="RANDOM", Q=None):
        self.type = type
        self.stats = copy.deepcopy(base_stats[type])
        self.attacks = ["TACKLE", base_attacks[type]]
        self.brain = brain
        self.Q = Q

    def attack(self, enemy, which_attack):
        this_attack = attacks[which_attack]
        # Base
        damage = this_attack["POWER"] * self.stats["ATT"] / enemy.stats["DEF"]
        # Random
        damage *= random.randrange(85, 101) / 100
        # STAB
        damage *= 1.5 if this_attack["TYPE"] == self.type else 1
        # Effectiveness
#        damage *= 0.5 if this_attack["TYPE"] == enemy.type else 1
        # Critical
        # Base, ensure never 0 damage
        damage += 2
        # Deal the damage
        lost = enemy.take_damage(damage)
        return(lost)

    def take_damage(self, damage):
        lost = damage / base_stats[self.type]["HP"]
        self.stats["HP"] -= damage
        if (self.stats["HP"] < 0):
            self.stats["HP"] = 0
        return(lost)

    def choose_attack(self, choice=None, state=None, epsilon=0):
        if choice is not None:
            return(choice)
        if (self.brain == "TACKLE"):
            return("TACKLE")
        elif (self.brain == "RANDOM"):
            return(random.choice(self.attacks))
        elif (self.brain == "Q"):
            if (random.random() > epsilon):
                print(state)
                # if both legal moves are negative, it chooses an illegal move.
                possible_Q = { attack: self.Q[state][attack] for attack in self.attacks}
                return(max(possible_Q, key=possible_Q.get))
            else:
                return(random.choice(self.attacks))
        # add an else???


if __name__ == '__main__':
    p = Pokemon("GREEN")
    print(p)
