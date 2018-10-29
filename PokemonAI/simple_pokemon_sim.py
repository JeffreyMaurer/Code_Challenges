from Pokemon import *
import random

# 2 Pokemon, just tackle
print("tackle")
p = Pokemon("GREEN")
q = Pokemon("RED")
while (True):
    if (p.stats["SPD"] > q.stats["SPD"]):
        p.attack(q, "TACKLE")
        if q.stats["HP"] == 0:
            break
        q.attack(p, "TACKLE")
        if p.stats["HP"] == 0:
            break
    else:
        q.attack(p, "TACKLE")
        if p.stats["HP"] == 0:
            break
        p.attack(q, "TACKLE")
        if q.stats["HP"] == 0:
            break

print(p.stats)
print(q.stats)

# 2 Pokemon, just tackle, pointers
print("pointers")
p = Pokemon("GREEN")
q = Pokemon("RED")
faster, slower = (p, q) if (p.stats["SPD"] > q.stats["SPD"]) else (q, p)
while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
    faster.attack(slower, "TACKLE")
    if slower.stats["HP"] == 0:
        break
    slower.attack(faster, "TACKLE")

print(p.stats)
print(q.stats)

# 2 Pokemon, prechosen, both choose moves by random
print("random")
p = Pokemon("GREEN")
q = Pokemon("RED")
faster, slower = (p, q) if (p.stats["SPD"] > q.stats["SPD"]) else (q, p)
while (p.stats["HP"] > 0 and q.stats["HP"] > 0):
    fa, sa = (random.choice(faster.attacks), random.choice(slower.attacks))
    faster.attack(slower, fa)
    if slower.stats["HP"] == 0:
        break
    slower.attack(faster, sa)

print(p.stats)
print(q.stats)
